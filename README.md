<div align="center">
 <h1> 🎭 Interactive Story Generator</h1>
 <h3>Create Engaging AI-Powered Stories</h3>
 <img src="https://img.shields.io/badge/python-%3E%3D3.8-blue?style=flat&logo=python&logoColor=white"/>
 <img src="https://img.shields.io/badge/gradio-4.44.1-orange?style=flat&logo=gradio&logoColor=white"/>
 <img src="https://img.shields.io/badge/huggingface-0.25.2-yellow?style=flat&logo=huggingface&logoColor=white"/>
 <img src="https://img.shields.io/badge/license-Apache%202.0-brightgreen?style=flat"/>
</div>
<br/>

![screencapture-huggingface-co-spaces-ChanMeng666-interactive-story-generator-2024-12-12-13_45_22](https://github.com/user-attachments/assets/2e03f96c-8ce8-4ab2-aed2-75dd31e61eb3)

![screencapture-huggingface-co-spaces-ChanMeng666-interactive-story-generator-2024-12-12-13_45_41](https://github.com/user-attachments/assets/f1e329be-59dc-40cf-80e7-54c3eb9303e7)


# ✨ Features

### 🤝 Interactive AI Collaboration
Craft unique stories through natural dialogue with an advanced AI storyteller. Our system adapts to your input and helps develop engaging narratives that bring your ideas to life.

### 🎨 Rich Story Customization
Choose from diverse themes like adventure, mystery, romance, and more. Fine-tune your narrative style from fantasy to sci-fi, ensuring each story matches your creative vision.

### 👤 Dynamic Character Creation
Build compelling characters using expert-designed templates or create your own from scratch. Develop rich personalities that drive your story forward.

### ⚙️ Advanced Story Controls
Take full control of your narrative with adjustable parameters for creativity, coherence, and story length. Shape the storytelling experience exactly how you want it.

### 💾 Story Management
Never lose your creative work with built-in story saving and management features. Easily revisit and continue your narratives at any time.

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/python-%3E%3D3.8-blue?style=for-the-badge&logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/gradio-4.44.1-orange?style=for-the-badge&logo=gradio&logoColor=white)
![HuggingFace](https://img.shields.io/badge/huggingface-0.25.2-yellow?style=for-the-badge&logo=huggingface&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-2.2.1-red?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/transformers-4.38.2-purple?style=for-the-badge&logo=transformers&logoColor=white)

## 🚀 Getting Started

<details>
<summary>Installation Guide</summary>

### Prerequisites
- Python 3.8 or higher
- A Hugging Face API token

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/ChanMeng666/interactive-story-generator.git
cd interactive-story-generator
```

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

</details>

## 💡 Usage Guide

<details>
<summary>How to Create Your Story</summary>

1. **Choose Your Theme**
   - Select from multiple story themes and styles
   - Customize the narrative tone to match your vision

2. **Set Up Characters**
   - Pick from character templates or create custom ones
   - Define personality traits and backgrounds

3. **Begin Your Story**
   - Describe your initial scene
   - Click "Start Story" to begin the AI collaboration

4. **Develop the Narrative**
   - Interact with the AI to progress the story
   - Guide the plot development through natural dialogue

5. **Fine-tune Generation**
   - Adjust creativity settings (Temperature: 0.1-2.0)
   - Control story length (Max Tokens: 64-1024)
   - Modify narrative diversity (Top-p: 0.1-1.0)

6. **Save Your Work**
   - Export your story at any time
   - Continue previous stories seamlessly

</details>

## 🔧 Advanced Configuration

<details>
<summary>Generation Parameters</summary>

### Temperature
Controls creativity level (0.1-2.0):
- Higher values (>1.0) produce more creative, varied output
- Lower values (<1.0) generate more focused, consistent content

### Max Tokens
Adjusts response length (64-1024):
- Higher values allow for longer story segments
- Lower values create more concise responses

### Top-p (Nucleus Sampling)
Fine-tunes output diversity (0.1-1.0):
- Higher values increase response variety
- Lower values make outputs more deterministic

</details>

## 📝 License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Built with [Gradio](https://gradio.app)
- Powered by [Hugging Face](https://huggingface.co)
- Uses Meta's Llama model

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
