import gradio as gr
from huggingface_hub import InferenceClient
import time
from typing import Optional, Generator
import logging
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


STORY_THEMES = [
    "冒险",
    "神秘",
    "浪漫",
    "历史",
    "日常",
    "童话"
]

CHARACTER_TEMPLATES = {
    "冒险家": "一个勇敢无畏的探险家，热爱冒险与挑战。",
    "侦探": "一个敏锐细心的侦探，善于观察和推理。",
    "艺术家": "一个富有创造力的艺术家，对美有独特的见解。",
    "科学家": "一个求知若渴的科学家，致力于探索未知。",
    "普通人": "一个平凡但内心丰富的普通人。"
}



# 初始化故事生成器的系统提示

STORY_SYSTEM_PROMPT = """你是一个专业的故事生成器。你的任务是根据用户提供的设定和实时输入，生成连贯且引人入胜的故事。

关键要求：
1. 故事必须具有连续性，每次回应都要基于之前的所有情节发展
2. 认真分析对话历史，保持人物性格、情节走向的一致性
3. 当用户补充新的细节或提供新的发展方向时，自然地将其整合到现有故事中
4. 注意因果关系，确保每个情节的发生都有合理的铺垫和解释
5. 通过环境描写、人物对话等手法，让故事更加生动
6. 在故事发展的关键节点，可以给出一些暗示，引导用户参与情节推进

你不应该：
1. 重新开始新的故事
2. 忽视之前提到的重要情节或细节
3. 生成与已建立设定相矛盾的内容
4. 突兀地引入未经铺垫的重大转折

请记住：你正在创作一个持续发展的故事，而不是独立的片段。"""


STORY_STYLES = [
    "奇幻",
    "科幻",
    "悬疑",
    "冒险",
    "爱情",
    "恐怖"
]

MAX_RETRIES = 3
RETRY_DELAY = 2

def create_client() -> InferenceClient:
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN 环境变量未设置")
    return InferenceClient(
        "HuggingFaceH4/zephyr-7b-beta",
        token=hf_token
    )

def generate_story(
    scene: str,
    style: str,
    theme: str,
    character_desc: str,
    history: list = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.95,
) -> Generator[str, None, None]:
    """
    生成连续性的故事情节
    """
    if history is None:
        history = []
        
    # 构建上下文摘要
    context_summary = ""
    story_content = []
    
    # 提取之前的故事内容
    for msg in history:
        if msg["role"] == "assistant":
            story_content.append(msg["content"])
    
    if story_content:
        context_summary = "\n".join([
            "已经发生的故事情节：",
            "---",
            "\n".join(story_content),
            "---"
        ])
    
    # 根据是否有历史记录使用不同的提示模板
    if not history:
        # 首次生成，使用完整设定
        prompt = f"""
        请基于以下设定开始讲述一个故事：
        
        风格：{style}
        主题：{theme}
        角色：{character_desc}
        初始场景：{scene}
        
        请从这个场景开始，展开故事的开端。注意为后续发展留下铺垫。
        """
    else:
        # 后续生成，侧重情节延续
        prompt = f"""
        {context_summary}
        
        故事设定提醒：
        - 风格：{style}
        - 主题：{theme}
        - 主要角色：{character_desc}
        
        用户新的输入：{scene}
        
        请基于以上已发生的情节和用户新的输入，自然地继续发展故事。注意：
        1. 新的发展必须与之前的情节保持连贯
        2. 合理化用户提供的新元素
        3. 注意人物性格的一致性
        4. 为后续发展留下可能性
        
        继续讲述：
        """
    
    messages = [
        {"role": "system", "content": STORY_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
        
    try:
        client = create_client()
        response = ""
        
        for message in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            if hasattr(message.choices[0].delta, 'content'):
                token = message.choices[0].delta.content
                if token is not None:
                    response += token
                    yield response
    except Exception as e:
        logger.error(f"生成故事时发生错误: {str(e)}")
        yield f"抱歉，生成故事时遇到了问题：{str(e)}\n请稍后重试。"



def summarize_story_context(history: list) -> str:
    """
    总结当前的故事上下文，用于辅助生成
    """
    if not history:
        return ""
    
    summary_parts = []
    key_elements = {
        "characters": set(),  # 出场人物
        "locations": set(),   # 场景地点
        "events": [],         # 关键事件
        "objects": set()      # 重要物品
    }
    
    for msg in history:
        content = msg.get("content", "")
        # TODO: 这里可以添加更复杂的NLP处理来提取关键信息
        # 当前使用简单的文本累加
        if content:
            summary_parts.append(content)
    
    return "\n".join(summary_parts)



# 创建故事生成器界面

def create_demo():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎭 互动式故事生成器
            让AI为您创造独特的故事体验。您可以选择故事风格、主题,添加角色设定,
            然后描述一个场景开始您的故事。与AI互动来继续发展故事情节!
            """
        )
        
        with gr.Tabs():
            # 故事创作标签页
            with gr.Tab("✍️ 故事创作"):
                with gr.Row(equal_height=True):
                    # 左侧控制面板
                    with gr.Column(scale=1):
                        with gr.Group():
                            style_select = gr.Dropdown(
                                choices=STORY_STYLES,
                                value="奇幻",
                                label="选择故事风格",
                                info="选择一个整体风格来定义故事的基调"
                            )
                            
                            theme_select = gr.Dropdown(
                                choices=STORY_THEMES,
                                value="冒险",
                                label="选择故事主题",
                                info="选择故事要重点表现的主题元素"
                            )
                        
                        with gr.Group():
                            gr.Markdown("### 👤 角色设定")
                            character_select = gr.Dropdown(
                                choices=list(CHARACTER_TEMPLATES.keys()),
                                value="冒险家",
                                label="选择角色模板",
                                info="选择一个预设的角色类型,或自定义描述"
                            )
                            
                            character_desc = gr.Textbox(
                                lines=3,
                                value=CHARACTER_TEMPLATES["冒险家"],
                                label="角色描述",
                                info="描述角色的性格、背景、特点等"
                            )
                            
                        with gr.Group():
                            scene_input = gr.Textbox(
                                lines=3,
                                placeholder="在这里描述故事发生的场景、环境、时间等...",
                                label="场景描述",
                                info="详细的场景描述会让故事更加生动"
                            )
                        
                        with gr.Row():
                            submit_btn = gr.Button("✨ 开始故事", variant="primary", scale=2)
                            clear_btn = gr.Button("🗑️ 清除对话", scale=1)
                            save_btn = gr.Button("💾 保存故事", scale=1)
                    
                    # 右侧对话区域
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="故事对话",
                            height=600,
                            show_label=True
                        )
                        
                        status_msg = gr.Markdown("")
            
            # 设置标签页
            with gr.Tab("⚙️ 高级设置"):
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=0.7,
                                step=0.1,
                                label="创意度(Temperature)",
                                info="较高的值会让故事更有创意但可能不够连贯"
                            )
                            
                            max_tokens = gr.Slider(
                                minimum=64,
                                maximum=1024,
                                value=512,
                                step=64,
                                label="最大生成长度",
                                info="控制每次生成的文本长度"
                            )
                            
                            top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.95,
                                step=0.05,
                                label="采样范围(Top-p)",
                                info="控制词语选择的多样性"
                            )
        
        # 帮助信息
        with gr.Accordion("📖 使用帮助", open=False):
            gr.Markdown(
                """
                ## 如何使用故事生成器
                1. 选择故事风格和主题来确定故事的整体基调
                2. 选择预设角色模板或自定义角色描述
                3. 描述故事发生的场景和环境
                4. 点击"开始故事"生成开篇
                5. 继续输入内容与AI交互,推进故事发展
                
                ## 小提示
                - 详细的场景和角色描述会让生成的故事更加丰富
                - 可以使用"保存故事"功能保存精彩的故事情节
                - 在设置中调整参数可以影响故事的创意程度和连贯性
                - 遇到不满意的情节可以使用"清除对话"重新开始
                
                ## 参数说明
                - 创意度: 控制故事的创意程度,值越高创意性越强
                - 采样范围: 控制用词的丰富程度,值越高用词越多样
                - 最大长度: 控制每次生成的文本长度
                """
            )
        
        # 更新角色描述
        def update_character_desc(template):
            return CHARACTER_TEMPLATES[template]
            
        character_select.change(
            update_character_desc,
            character_select,
            character_desc
        )
        
        # 保存故事对话
        save_btn.click(
            save_story,
            chatbot,
            status_msg,
        )
        
        # 用户输入处理
        def user_input(user_message, history):
            """
            处理用户输入
            Args:
                user_message: 用户输入的消息
                history: 聊天历史记录 [(user_msg, bot_msg), ...]
            """
            if history is None:
                history = []
            history.append([user_message, None])  # 添加用户消息,bot消息暂时为None
            return "", history
            
        # AI响应处理
        def bot_response(history, style, theme, character_desc, temperature, max_tokens, top_p):
            """
            生成AI响应
            Args:
                history: 聊天历史记录 [(user_msg, bot_msg), ...]
                style: 故事风格
                theme: 故事主题
                character_desc: 角色描述
                temperature: 生成参数
                max_tokens: 生成参数
                top_p: 生成参数
            """
            try:

                # 获取用户的最后一条消息
                user_message = history[-1][0] 
                
                # 转换历史记录格式以传递给generate_story
                message_history = []
                for user_msg, bot_msg in history[:-1]:  # 不包括最后一条
                    if user_msg:
                        message_history.append({"role": "user", "content": user_msg})
                    if bot_msg:
                        message_history.append({"role": "assistant", "content": bot_msg})
                        
                # 开始生成故事
                current_response = ""
                for text in generate_story(
                    user_message,
                    style,
                    theme,
                    character_desc,
                    message_history,
                    temperature,
                    max_tokens,
                    top_p
                ):
                    current_response = text
                    history[-1][1] = current_response  # 更新最后一条消息的bot回复
                    yield history
                    
            except Exception as e:
                logger.error(f"处理响应时发生错误: {str(e)}")
                error_msg = f"抱歉，生成故事时遇到了问题。请稍后重试。"
                history[-1][1] = error_msg
                yield history

        
        # 清除对话
        def clear_chat():
            return [], ""
        
        # 绑定事件
        scene_input.submit(
            user_input,
            [scene_input, chatbot],
            [scene_input, chatbot]
        ).then(
            bot_response,
            [chatbot, style_select, theme_select, character_desc, temperature, max_tokens, top_p],
            chatbot
        )
        
        submit_btn.click(
            user_input,
            [scene_input, chatbot],
            [scene_input, chatbot]
        ).then(
            bot_response,
            [chatbot, style_select, theme_select, character_desc, temperature, max_tokens, top_p],
            chatbot
        )
        
        clear_btn.click(
            clear_chat,
            None,
            [chatbot, status_msg],
        )
        
        return demo


def save_story(chatbot):
    """保存故事对话记录"""
    if not chatbot:
        return "故事为空,无法保存"
        
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"stories/story_{timestamp}.txt"
    
    os.makedirs("stories", exist_ok=True)
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for user_msg, bot_msg in chatbot:
                if user_msg:
                    f.write(f"用户: {user_msg}\n")
                if bot_msg:
                    f.write(f"AI: {bot_msg}\n\n")
        return f"故事已保存至 {filename}"
    except Exception as e:
        return f"保存失败: {str(e)}"



if __name__ == "__main__":
    demo = create_demo()
    demo.queue().launch(
        # server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

