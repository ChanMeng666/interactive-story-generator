import gradio as gr
from huggingface_hub import InferenceClient
import time
from typing import Optional, Generator
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


STORY_THEMES = [
    "Adventure",
    "Mystery",
    "Romance",
    "Historical",
    "Slice of Life",
    "Fairy Tale"
]

CHARACTER_TEMPLATES = {
    "Adventurer": "A brave and fearless explorer who loves adventure and challenges.",
    "Detective": "A keen and observant detective skilled in observation and deduction.",
    "Artist": "A creative artist with unique perspectives on beauty.",
    "Scientist": "A curious scientist dedicated to exploring the unknown.",
    "Ordinary Person": "An ordinary person with a rich inner world."
}

# Initialize story generator system prompt
STORY_SYSTEM_PROMPT = """You are a professional story generator. Your task is to generate coherent and engaging stories based on user settings and real-time input.

Key requirements:
1. The story must maintain continuity, with each response building upon all previous plot developments
2. Carefully analyze dialogue history to maintain consistency in character personalities and plot progression
3. Naturally integrate new details or development directions when provided by the user
4. Pay attention to cause and effect, ensuring each plot point has reasonable setup and explanation
5. Make the story more vivid through environmental descriptions and character dialogues
6. At key story points, provide hints to guide user participation in plot progression

You should not:
1. Start a new story
2. Ignore previously mentioned important plots or details
3. Generate content that contradicts established settings
4. Introduce major turns without proper setup

Remember: You are creating an ongoing story, not independent fragments."""


STORY_STYLES = [
    "Fantasy",
    "Science Fiction",
    "Mystery",
    "Adventure",
    "Romance",
    "Horror"
]

MAX_RETRIES = 3
RETRY_DELAY = 2

def create_client() -> InferenceClient:
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")
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
    Generate continuous story plot
    """
    if history is None:
        history = []
        
    # Build context summary
    context_summary = ""
    story_content = []
    
    # Extract previous story content
    for msg in history:
        if msg["role"] == "assistant":
            story_content.append(msg["content"])
    
    if story_content:
        context_summary = "\n".join([
            "Previously in the story:",
            "---",
            "\n".join(story_content),
            "---"
        ])
    
    # Use different prompt templates based on whether there's history
    if not history:
        # First generation, use complete settings
        prompt = f"""
        Please start a story based on the following settings:
        
        Style: {style}
        Theme: {theme}
        Character: {character_desc}
        Initial Scene: {scene}
        
        Please begin from this scene and set up the story's opening. Leave room for future developments.
        """
    else:
        # Subsequent generation, focus on plot continuation
        prompt = f"""
        {context_summary}
        
        Story settings reminder:
        - Style: {style}
        - Theme: {theme}
        - Main Character: {character_desc}
        
        User's new input: {scene}
        
        Please continue the story based on the previous plot and user's new input. Note:
        1. New developments must maintain continuity with previous plot
        2. Rationalize new elements provided by the user
        3. Maintain consistency in character personalities
        4. Leave possibilities for future developments
        
        Continue the story:
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
        logger.error(f"Error occurred while generating story: {str(e)}")
        yield f"Sorry, encountered an error while generating the story: {str(e)}\nPlease try again later."

def summarize_story_context(history: list) -> str:
    """
    Summarize current story context for generation assistance
    """
    if not history:
        return ""
    
    summary_parts = []
    key_elements = {
        "characters": set(),  # Characters appeared
        "locations": set(),   # Scene locations
        "events": [],         # Key events
        "objects": set()      # Important items
    }
    
    for msg in history:
        content = msg.get("content", "")
        # TODO: More complex NLP processing can be added here to extract key information
        # Currently using simple text accumulation
        if content:
            summary_parts.append(content)
    
    return "\n".join(summary_parts)

# Create story generator interface
def create_demo():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎭 Interactive Story Generator
            Let AI create a unique storytelling experience for you. Choose your story style, theme, add character settings,
            then describe a scene to start your story. Interact with AI to continue developing the plot!
            """
        )
        
        with gr.Tabs():
            # Story Creation Tab
            with gr.Tab("✍️ Story Creation"):
                with gr.Row(equal_height=True):
                    # Left Control Panel
                    with gr.Column(scale=1):
                        with gr.Group():
                            style_select = gr.Dropdown(
                                choices=STORY_STYLES,
                                value="Fantasy",
                                label="Choose Story Style",
                                info="Select an overall style to define the story's tone"
                            )
                            
                            theme_select = gr.Dropdown(
                                choices=STORY_THEMES,
                                value="Adventure",
                                label="Choose Story Theme",
                                info="Select the main thematic elements to focus on"
                            )
                        
                        with gr.Group():
                            gr.Markdown("### 👤 Character Settings")
                            character_select = gr.Dropdown(
                                choices=list(CHARACTER_TEMPLATES.keys()),
                                value="Adventurer",
                                label="Select Character Template",
                                info="Choose a preset character type or customize description"
                            )
                            
                            character_desc = gr.Textbox(
                                lines=3,
                                value=CHARACTER_TEMPLATES["Adventurer"],
                                label="Character Description",
                                info="Describe character's personality, background, traits, etc."
                            )
                            
                        with gr.Group():
                            scene_input = gr.Textbox(
                                lines=3,
                                placeholder="Describe the scene, environment, time, etc. here...",
                                label="Scene Description",
                                info="Detailed scene description will make the story more vivid"
                            )
                        
                        with gr.Row():
                            submit_btn = gr.Button("✨ Start Story", variant="primary", scale=2)
                            clear_btn = gr.Button("🗑️ Clear Chat", scale=1)
                            save_btn = gr.Button("💾 Save Story", scale=1)
                    
                    # Right Chat Area
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="Story Dialogue",
                            height=600,
                            show_label=True
                        )
                        
                        status_msg = gr.Markdown("")
            
            # Settings Tab
            with gr.Tab("⚙️ Advanced Settings"):
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=0.7,
                                step=0.1,
                                label="Creativity (Temperature)",
                                info="Higher values make story more creative but potentially less coherent"
                            )
                            
                            max_tokens = gr.Slider(
                                minimum=64,
                                maximum=1024,
                                value=512,
                                step=64,
                                label="Maximum Generation Length",
                                info="Control the length of each generated text"
                            )
                            
                            top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.95,
                                step=0.05,
                                label="Sampling Range (Top-p)",
                                info="Control the diversity of word choice"
                            )
        
        # Help Information
        with gr.Accordion("📖 Usage Guide", open=False):
            gr.Markdown(
                """
                ## How to Use the Story Generator
                1. Choose story style and theme to set the overall tone
                2. Select a preset character template or customize character description
                3. Describe the story's scene and environment
                4. Click "Start Story" to generate the opening
                5. Continue inputting content to interact with AI and advance the story
                
                ## Tips
                - Detailed scene and character descriptions will make the generated story richer
                - Use the "Save Story" function to save memorable story plots
                - Adjust parameters in settings to affect story creativity and coherence
                - Use "Clear Chat" to start over if you're not satisfied with the plot
                
                ## Parameter Explanation
                - Creativity: Controls the story's creativity level, higher values increase creativity
                - Sampling Range: Controls vocabulary richness, higher values increase word diversity
                - Maximum Length: Controls the length of each generated text
                """
            )
        
        # Update character description
        def update_character_desc(template):
            return CHARACTER_TEMPLATES[template]
            
        character_select.change(
            update_character_desc,
            character_select,
            character_desc
        )
        
        # Save story dialogue
        save_btn.click(
            save_story,
            inputs=[
                chatbot,
                style_select,
                theme_select, 
                character_desc
            ],
            outputs=status_msg
        )
        
        # User input processing
        def user_input(user_message, history):
            """
            Process user input
            Args:
                user_message: User's input message
                history: Chat history [(user_msg, bot_msg), ...]
            """
            if history is None:
                history = []
            history.append([user_message, None])  # Add user message, bot message temporarily None
            return "", history
            
        # AI response processing
        def bot_response(history, style, theme, character_desc, temperature, max_tokens, top_p):
            """
            Generate AI response
            Args:
                history: Chat history [(user_msg, bot_msg), ...]
                style: Story style
                theme: Story theme
                character_desc: Character description
                temperature: Generation parameter
                max_tokens: Generation parameter
                top_p: Generation parameter
            """
            try:
                # Get user's last message
                user_message = history[-1][0] 
                
                # Convert history format for generate_story
                message_history = []
                for user_msg, bot_msg in history[:-1]:  # Excluding the last one
                    if user_msg:
                        message_history.append({"role": "user", "content": user_msg})
                    if bot_msg:
                        message_history.append({"role": "assistant", "content": bot_msg})
                        
                # Start generating story
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
                    history[-1][1] = current_response  # Update bot reply for the last message
                    yield history
                    
            except Exception as e:
                logger.error(f"Error occurred while processing response: {str(e)}")
                error_msg = f"Sorry, encountered an error while generating the story. Please try again later."
                history[-1][1] = error_msg
                yield history

        
        # Clear chat
        def clear_chat():
            return [], ""
        
        # Bind events
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


def save_story(chatbot, style=None, theme=None, character_desc=None):
    """
    Save story dialogue record with metadata
    Args:
        chatbot: Chat history containing user and AI messages
        style: Story style selected by user
        theme: Story theme selected by user 
        character_desc: Character description
    Returns:
        Status message indicating success or failure
    """
    if not chatbot:
        return "Story is empty, cannot save"
        
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create stories directory if it doesn't exist
    # Use absolute path for Hugging Face Space
    stories_dir = os.path.join(os.getcwd(), "stories")
    os.makedirs(stories_dir, exist_ok=True)
    
    filename = os.path.join(stories_dir, f"story_{timestamp}.txt")
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            # Write header with metadata
            f.write("=== Interactive Story ===\n")
            f.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if style:
                f.write(f"Style: {style}\n")
            if theme:
                f.write(f"Theme: {theme}\n")
            if character_desc:
                f.write(f"Character: {character_desc}\n")
            
            f.write("\n=== Story Content ===\n\n")
            
            # Write conversation
            for i, (user_msg, ai_msg) in enumerate(chatbot, 1):
                f.write(f"--- Turn {i} ---\n")
                if user_msg:
                    f.write(f"User: {user_msg}\n")
                if ai_msg:
                    f.write(f"AI: {ai_msg}\n")
                f.write("\n")
                
        # Return success message with filename
        return gr.Markdown(f"✅ Story saved successfully to: {os.path.basename(filename)}")
        
    except Exception as e:
        logger.error(f"Error saving story: {str(e)}")
        return gr.Markdown(f"❌ Failed to save story: {str(e)}")

if __name__ == "__main__":
    demo = create_demo()
    demo.queue().launch(
        # server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

