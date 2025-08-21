# gguf-ai_model_ai-server-

namanmic_gguf Universal Server
A powerful, self-hosted AI server that provides OpenAI-compatible API endpoints using Mistral-7B model with advanced anti-detection capabilities.

Features
# anti ai detection response 
# index.html aichatbot using this api server 
OpenAI-Compatible API: Full compatibility with OpenAI's API structure
Advanced Anti-Detection: Built-in StealthWriter technology to humanize AI responses
Multi-Format Support: Process PDFs, images, and text files
Session Management: Maintain conversation context and file references
API Key Management: Secure access control with admin capabilities
Self-Hosted: Complete privacy and control over your AI interactions
Installation

Clone the repository
bash
git clone https://github.com/nksdev/gguf-ai_model_ai-server-
cd gguf-ai_model_ai-server-
Install dependencies
bash
pip install -r requirements.txt
Download the Mistral model
bash
# Create models directory
mkdir -p models

# Download the model (replace with your actual model path)

#download model from https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_0.gguf?download=true

wget -O models/mistral-7b-instruct-v0.2.Q5_0.gguf <model-download-url>
Install additional dependencies
bash
# For OCR functionality
sudo apt-get install tesseract-ocr

# For spaCy model
python -m spacy download en_core_web_sm
Configuration

The server can be configured by modifying the config dictionary in app.py:

python
config = {
    'models_folder': 'models',
    'model_file': 'mistral-7b-instruct-v0.2.Q5_0.gguf',
    'api_keys_file': 'api_keys.json',
    'upload_folder': 'uploads',
    'allowed_extensions': {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx'},
    'host': '0.0.0.0',
    'port': 5010,
    'debug': False,
    'n_ctx': 8192,
    'n_threads': 8,
    'n_gpu_layers': 32,
    'auto_humanize': True  # Enable/disable automatic humanization
}
Usage

Starting the Server

bash
python app.py
The server will start on http://localhost:5010 with a web interface available at the root URL.

API Authentication

All API endpoints require authentication using API keys. A default admin key is automatically generated on first run.

Get Your API Key

Check the server logs on first run to see the auto-generated admin API key, or create new keys using the admin endpoints.

API Endpoints

1. Chat Completions (OpenAI Compatible)

bash
curl -X POST "http://localhost:5010/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b-instruct",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
2. Text Completions (OpenAI Compatible)

bash
curl -X POST "http://localhost:5010/v1/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b-instruct",
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.7
  }'
3. File Upload

bash
curl -X POST "http://localhost:5010/v1/upload" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@document.pdf"
4. Humanize Text (Anti-Detection)

bash
curl -X POST "http://localhost:5010/v1/plag" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "AI-generated text to humanize",
    "format": "key_points",
    "output_format": "markdown"
  }'
5. Session Management

bash
# List files in current session
curl -X GET "http://localhost:5010/v1/session/files" \
  -H "Authorization: Bearer YOUR_API_KEY"

# Clear current session
curl -X POST "http://localhost:5010/v1/session/clear" \
  -H "Authorization: Bearer YOUR_API_KEY"
6. Admin Endpoints

bash
# List all API keys (admin only)
curl -X GET "http://localhost:5010/admin/api-keys" \
  -H "Authorization: Bearer ADMIN_API_KEY"

# Create new API key
curl -X POST "http://localhost:5010/admin/api-keys" \
  -H "Authorization: Bearer ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "user1", "permissions": ["read"]}'
Web Interface

The server includes a web-based chat interface accessible at http://localhost:5010 that provides:

Interactive chat with the AI model
File upload capabilities
Response formatting options
Session management
Technical Approach

Architecture

The server is built with a modular architecture:

Flask Application: Core web server with rate limiting and CORS
Mistral Model Wrapper: Handles model loading and inference
File Processor: Extracts text from PDFs and images using OCR
Session Manager: Maintains user sessions and file contexts
API Key Manager: Handles authentication and authorization
StealthWriter: Advanced anti-detection text transformation
Anti-Detection Technology

The StealthWriter module uses multiple techniques to make AI text undetectable:

Contextual Synonym Replacement: Replaces AI-typical words with human alternatives
Structural Variation: Changes sentence structure and paragraph organization
Human Pattern Injection: Adds hedging phrases, interjections, and casual language
Formatting Diversity: Introduces human-like formatting variations
Error Introduction: Adds minor errors that humans make but AI typically doesn't
Workflow

Request Processing: API requests are authenticated and validated
Context Building: File references are extracted and included in context
Prompt Construction: Messages are formatted for the Mistral model
Response Generation: The model generates a completion
Humanization: Responses are transformed to avoid AI detection
Session Update: Conversation history is maintained for context
Project Ideas

This server can be used to build:

Academic Writing Assistant: Help with research papers and essays
Content Creation Platform: Generate human-like content for blogs and marketing
Document Analysis Tool: Extract insights from uploaded documents
Educational Tutor: Provide explanations and answers to student questions
Code Explanation Tool: Analyze and explain code snippets
Troubleshooting

Common Issues

Model not loading: Ensure the model file is in the models directory
OCR not working: Install Tesseract OCR and ensure images are clear
Memory errors: Reduce n_ctx or n_gpu_layers in configuration
API key issues: Check server logs for the auto-generated admin key
Performance Tips

Adjust n_gpu_layers based on your GPU VRAM
Reduce n_ctx for less memory usage (but shorter context)
Use n_threads to optimize for your CPU core count
Enable auto_humanize only when anti-detection is needed for performance
License

This project is provided for educational and research purposes. Please ensure compliance with the Mistral model's license terms and use responsibly.

Support

For issues and questions:

Check the server logs for error messages
Verify all dependencies are installed correctly
Ensure sufficient system resources (RAM, VRAM) are available
