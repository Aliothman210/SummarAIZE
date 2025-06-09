# SummarAIze 📝  
**AI-Powered Multilingual Document Summarization**

SummarAIze is an intelligent summarization tool that leverages state-of-the-art NLP models to generate concise summaries from text, PDFs, Word documents, images (OCR), and spreadsheets.

Designed for efficiency and multilingual support, it processes content in **Arabic**, **English**, and **7+ other languages**, while preserving formatting and context.

---

## 🚀 Key Features

- ✅ **Multilingual Support**  
  Powered by Hugging Face's mT5 model for Arabic, English, and more.

- ✅ **Document Processing**  
  Extracts text from:
  - PDFs  
  - Word documents  
  - Images (OCR)  
  - CSV / Excel files

- ✅ **Customizable Outputs**  
  Adjust summary length and download results in:
  - TXT  
  - PDF  
  - Word  
  - Excel

- ✅ **User-Friendly**  
  Streamlit-based UI with responsive design and GPU acceleration support.

---

## 🛠 Technical Stack

- **Backend**: Python, PyTorch, Transformers (Hugging Face)  
- **OCR**: Tesseract (pytesseract)  
- **Frontend**: Streamlit  
- **Deployment**: Streamlit Cloud (with CI/CD via GitHub)  

---

## 🎯 Use Cases

- Research paper summarization  
- Business report condensation  
- Multilingual content analysis  

Optimized for both **CPU** and **GPU** (NVIDIA CUDA) environments.

---

## ⚙️ How to Deploy

```bash
git clone https://github.com/your-username/SummarAIze.git
pip install -r requirements.txt
streamlit run main.py
