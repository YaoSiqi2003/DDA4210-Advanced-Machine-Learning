from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from fastapi.responses import StreamingResponse
import os
import json
import requests
import traceback
from requests.exceptions import RequestException
import sys
import json
import traceback
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import urllib3
import certifi
from openai import OpenAI
from model_manager.model_manager import ModelManager
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

EMOTION_LIST = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
PROMPT = "Given a conversation segment without final utterance, identify the index number and speaker of the final utterance. Then, determine the speaker's previous emotion and the counterpart's emotion from their most recent utterance, selecting from the following list: {}. Next, determine whether the speaker's emotion will shift between their previous statement and the final statement, and whether the speaker's final emotion will be same with the counterpart's emotion. Afterward, predict the speaker's emotion in the final statement, selecting from the same list. Identify the cause of any emotion shift by directly citing the most significant sentence that triggered the change, and provide reasoning for why the identified cause led to the emotion change. Generate the speaker's final utterance based on the preceding conversation, the predicted emotion, and the analysis.\n{}"

# 全局变量存储模型和tokenizer
base_model_path = "model_path_to_DeepSeek-R1-Distill-Qwen-7B"
lora_path = "path_to_lora"

client = OpenAI(
    api_key="your_api_key_in_volce_engine",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 加载模型并应用LoRA权重
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 加载LoRA权重
model = PeftModel.from_pretrained(model, lora_path)

print("Model Loaded Successfully !")

app = Flask(__name__)

# 添加以下定义
ASSISTANT_SPEAKER = "Chandler"
USER_SPEAKER = "Monica"
current_session_id = "session_1"  # 默认对话编号，可由前端动态设置

processed = []  # 改为列表，每个元素代表一个对话

chat_index_counter = 0  # 聊天 index 递增

def safe_print(text):
    """适配控制台编码的输出方法"""
    encoded_text = text.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
    print(encoded_text)

def load_config():
    model_manager = ModelManager()
    model_config = {}
    
    model_type = "target_models"
    model_name = "DeepSeek-V3"
    
    try:
        model_config = model_manager.config[model_type][model_name]
        print(f"成功加载{model_name}推理模型配置")
    except Exception as e:
        raise(f"加载模型配置失败: {e}")
    
    return model_config

def set_client(model_config):
    client = OpenAI(
        api_key = model_config["api_key"],
        base_url = model_config["api_base_url"],
    )
    return client

def on_processed_updated(processed_data, temperature: float = 0.7, top_p: float = 1.0):
    """
    当 processed 数据更新时的回调函数
    提取当前会话的所有对话条目
    """
    # 准备请求数据，移除session_id
    model_prompt_data = {
        "Conversation": processed_data["Conversation"],
        "final_utterance_index": processed_data["final_utterance_index"],
        "final_speaker": processed_data["final_speaker"]
    }

    # 发送思考开始信号
    thinking_str = '<think>\n'
    yield f"data: {json.dumps({'thinking': thinking_str})}\n\n"
        
    prompt = PROMPT.format(EMOTION_LIST, model_prompt_data)
    # 准备输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generated_text = ""
    past_key_values = None
    input_ids = inputs.input_ids

    # 逐个token生成并流式输出
    for _ in range(2048):  # 设置最大生成长度限制
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=inputs.attention_mask,
                    use_cache=True
                )
                logits = outputs.logits
                past_key_values = outputs.past_key_values
            else:
                outputs = model(
                    input_ids=input_ids[:, -1:],
                    attention_mask=torch.ones(1, 1, device=model.device),
                    past_key_values=past_key_values,
                    use_cache=True
                )
                logits = outputs.logits
                past_key_values = outputs.past_key_values
        
        # 应用repetition_penalty=1.1
        repetition_penalty = 1.1
        if input_ids.shape[1] > 1:
            # 获取已生成的所有token的索引
            prev_tokens = input_ids[0].tolist()
            # 创建一个与logits形状相同的张量，值为1
            score = torch.ones_like(logits[:, -1])
            # 对已生成token的logits应用惩罚
            for token in set(prev_tokens):
                score[0, token] = repetition_penalty
            # 对已生成的token降低概率(repetition_penalty > 1)或提高概率(repetition_penalty < 1)
            logits[:, -1] = torch.div(logits[:, -1], score)
        
        # 采样下一个token
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # 解码当前token
        next_token_text = tokenizer.decode([next_token[0, 0].item()], skip_special_tokens=False)
        
        # 检查是否是特殊token或结束标记
        if next_token[0, 0].item() == tokenizer.eos_token_id or "<｜end▁of▁sentence｜>" in next_token_text:
            # 流式输出结束标志
            break
        
        # 跳过模型特定的格式标记，这些不应该在输出中显示
        skip_special_tokens = [
            "<｜User｜>", "<｜Assistant｜>", "<｜begin▁of▁sentence｜>", "<|EOT|>"
        ]
        
        if any(token in next_token_text for token in skip_special_tokens):
            continue
            
        generated_text += next_token_text
        print(next_token_text)
        
        # 流式输出思考内容
        yield f"data: {json.dumps({'thinking': next_token_text})}\n\n"

    # 在思考结束时发送特殊信号，表明已完成思考过程
    yield f"data: {json.dumps({'thinking_done': True, 'processed_result': generated_text})}\n\n"

    return generated_text

def on_response_updated(model_response):
    """
    二次处理模型响应
    """
    print("\n=== DEBUG: 进入 on_response_updated ===")
    print(f"DEBUG: model_response = {model_response}")

    try:
        if isinstance(model_response, dict):
            response_text = model_response.get('model_response', '')
        else:
            response_text = str(model_response)

        print(f"DEBUG: 处理后的响应文本: {response_text}")

        if not response_text or len(response_text.strip()) == 0:
            print("DEBUG: 空响应文本")
            yield f"data: {json.dumps({'error': '空响应文本'})}\n\n"
            return None

        EXTRACT_PROMPT = "Extract the content of final utterance from the analysis below. Only return the final utterance without any other text or explanation.\n\n{}"

        try:
            print("DEBUG: 开始调用 OpenAI API")
            stream = client.chat.completions.create(
                model="qwen-max",
                messages=[
                    {"role": "system", "content": EXTRACT_PROMPT},
                    {"role": "user", "content": response_text}
                ],
                stream=True
            )

            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    print(f"DEBUG: 收到响应片段: {content}")
                    # 发送实际内容块，用content字段
                    yield f"data: {json.dumps({'content': content})}\n\n"

            # 发送最终响应信号
            yield f"data: {json.dumps({'final_utterance': full_response})}\n\n"

            print(f"DEBUG: 完整最终响应: {full_response}")
            return full_response

        except Exception as e:
            print(f"DEBUG: 处理失败: {str(e)}")
            error_msg = f"[ERROR] Processing failed: {str(e)}"
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            return None

    except Exception as e:
        print(f"DEBUG: 处理错误: {str(e)}")
        error_msg = f"Processing error: {str(e)}"
        traceback.print_exc()
        yield f"data: {json.dumps({'error': error_msg})}\n\n"
        return None

@app.route('/')
def index():
    return render_template('chat3.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    global current_session_id
    user_message = request.form['message']
    session_id = request.form.get("conversationId")
    if session_id:
        current_session_id = session_id  # 自动同步前端 session_id

    # 返回普通响应（非流式），用于不支持流式响应的情况
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer f1fcb515-4526-422d-bdb6-f1be9664f805"
        }

        data = {
            "model": "deepseek-r1-250120",
            "messages": [
                {"role": "system", "content": "你是人工智能助手"},
                {"role": "user", "content": user_message}
            ]
        }

        print("发送请求到DeepSeek API (非流式)...")
        # 禁用SSL验证，用于测试
        response = requests.post(
            "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            headers=headers,
            json=data,
            timeout=30,
            verify=False  # 禁用SSL验证，仅用于测试
        )

        print(f"DeepSeek API (非流式) 响应状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            response_message = result['choices'][0]['message']['content']
        else:
            print(f"API 返回错误状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            response_message = f"API 返回错误状态码: {response.status_code}"

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"API 调用出错: {str(e)}")
        print(f"详细错误信息: {error_details}")
        response_message = f"抱歉，发生了错误: {str(e)}"

    response_data = {'message': response_message}
    json_str1 = json.dumps(response_data, ensure_ascii=False)
    return Response(json_str1, content_type='application/json; charset=utf-8')


@app.route('/stream_message', methods=['POST'])
def stream_message():
    global current_session_id
    # request.form: ImmutableMultiDict([('message', 'hello'), ('conversationId', '0')])
    user_message = request.form['message']
    session_id = request.form.get("conversationId")
    if session_id:
        current_session_id = session_id

    print(f"收到用户消息: {user_message}")

    def generate():
        try:
            try:
                # 检查是否已经存在当前会话
                current_conversation = None
                for conv in processed:
                    if conv.get("session_id") == current_session_id:
                        current_conversation = conv
                        break

                if current_conversation is None:
                    # 创建新的对话
                    current_conversation = {
                        "session_id": current_session_id,
                        "Conversation": [],
                        "final_utterance_index": 0,
                        "final_speaker": ASSISTANT_SPEAKER
                    }
                    processed.append(current_conversation)

                # 获取下一个消息的索引
                next_index = len(current_conversation["Conversation"])

                # 添加用户消息
                current_conversation["Conversation"].append({
                    "index": next_index,
                    "speaker": USER_SPEAKER,
                    "utterance": user_message
                })

                # 更新对话索引
                current_conversation["final_utterance_index"] = next_index + 1
                current_conversation["final_speaker"] = ASSISTANT_SPEAKER

                # 调用处理函数并生成结果
                # 第一阶段：模型生成分析
                processed_result = None
                for chunk in on_processed_updated(current_conversation):
                    yield chunk  # 直接传递chunk到前端
                    # 检查是否传递了processed结果
                    try:
                        chunk_data = json.loads(chunk.replace("data: ", ""))
                        if "processed_result" in chunk_data:
                            processed_result = chunk_data["processed_result"]
                    except:
                        pass
                
                # 确保我们获得了处理结果
                if processed_result is None:
                    print("警告: 无法从流中获取处理结果，尝试直接调用函数")
                    processed_result = on_processed_updated(current_conversation)
                    if hasattr(processed_result, '__iter__') and not isinstance(processed_result, str):
                        last_result = None
                        for item in processed_result:
                            last_result = item
                            try:
                                chunk_data = json.loads(item.replace("data: ", ""))
                                if "processed_result" in chunk_data:
                                    processed_result = chunk_data["processed_result"]
                                    break
                            except:
                                pass
                        if processed_result is None:
                            print("警告: 无法从生成器中获取处理结果，使用最后一个项目")
                            processed_result = last_result
                
                print(f"\n=== DEBUG: 第一阶段处理完成，结果: {processed_result} ===\n")
                    
                # 第二阶段：处理最终回复
                final_utterance = None
                if processed_result:
                    for chunk in on_response_updated(processed_result):
                        yield chunk  # 直接传递chunk到前端
                        # 尝试提取最终响应
                        try:
                            chunk_data = json.loads(chunk.replace("data: ", ""))
                            if "final_utterance" in chunk_data:
                                final_utterance = chunk_data["final_utterance"]
                        except:
                            pass

                print(f"\n=== DEBUG: 第二阶段处理完成，最终话语: {final_utterance} ===\n")

                # 添加助手消息到对话历史
                if final_utterance:
                    current_conversation["Conversation"].append({
                        "index": next_index + 1,
                        "speaker": ASSISTANT_SPEAKER,
                        "utterance": final_utterance
                    })
                elif processed_result:
                    # 如果没有提取到最终响应，使用处理结果
                    current_conversation["Conversation"].append({
                        "index": next_index + 1,
                        "speaker": ASSISTANT_SPEAKER,
                        "utterance": processed_result
                    })

                # 最后发送完成信号
                yield f"data: [DONE]\n\n"

            except Exception as e:
                print(f"请求处理错误: {e}")
                traceback.print_exc()
                yield f"data: {json.dumps({'error': f'请求处理错误: {str(e)}'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            error_details = traceback.format_exc()
            print(f"流式处理出错: {str(e)}")
            print(f"详细错误信息: {error_details}")
            yield f"data: {json.dumps({'error': f'流式处理出错: {str(e)}'}, ensure_ascii=False)}\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')


@app.route('/get_conversation_data', methods=['GET'])
def get_conversation_data():
    json_str = json.dumps(processed, ensure_ascii=False)
    return Response(json_str, content_type='application/json; charset=utf-8')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)