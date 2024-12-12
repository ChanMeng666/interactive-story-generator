# 🎭 Interactive Story Generator

An AI-powered interactive story generator that creates engaging narratives through collaboration between users and advanced language models. Built with Gradio and the Hugging Face Inference API, this application allows users to craft unique stories by providing settings, characters, and plot directions while the AI helps develop the narrative.

## ✨ Key Features

- **Interactive Storytelling**: Develop stories through natural dialogue with the AI
- **Multiple Story Themes**: Choose from various themes including adventure, mystery, romance, historical, slice of life, and fairy tales
- **Character Templates**: Select from predefined character archetypes or create custom characters
- **Flexible Story Styles**: Adapt the narrative style from fantasy to sci-fi, mystery to horror
- **Advanced Controls**: Fine-tune generation parameters like creativity and narrative coherence
- **Story Management**: Save and manage your generated stories

## 🛠️ Technology Stack

- Python 3.8+
- Gradio 4.44.1 (UI Framework)
- Hugging Face Hub 0.25.2 (Model Integration)
- PyTorch 2.2.1 (Deep Learning)
- Transformers 4.38.2 (NLP Models)
- Python-dotenv 1.0.0 (Environment Management)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- A Hugging Face API token

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ChanMeng666/interactive-story-generator.git
   cd interactive-story-generator

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file and add your Hugging Face API token:
   ```env
   HF_TOKEN=your_token_here
   ```

4. Run the application:
   ```bash
   python app.py
   ```

The application will be available at `http://localhost:7860`

## 💡 Usage

1. Select a story theme and style from the dropdown menus
2. Choose a character template or create your own
3. Describe the initial scene or setting
4. Click "Start Story" to begin
5. Continue developing the story through interaction with the AI
6. Use the advanced settings to fine-tune the narrative generation
7. Save your story when finished

## 🎛️ Advanced Settings

- **Temperature**: Control the creativity level of generated content (0.1-2.0)
- **Max Tokens**: Adjust the length of generated text (64-1024)
- **Top-p**: Fine-tune the diversity of generated content (0.1-1.0)

## 📝 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built using [Gradio](https://gradio.app)
- Powered by [Hugging Face](https://huggingface.co)
- Uses Meta's Llama model for text generation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.