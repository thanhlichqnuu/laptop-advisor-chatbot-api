# Laptop Advisor Chatbot API

This project provides a backend API using Python, FastAPI, and Langchain. 

## Prerequisites

Before you begin, ensure you have met the following requirements:

*   **Python 3.10:** This project requires Python 3.10 or higher. You can check your Python version by running:
    ```bash
    python --version 
    ```
*   **Gemini API Key:** You will need a Gemini API key to use the Gemini model. Obtain your key from [Google AI Studio](https://ai.google.dev/) and add it to the `.env` file as shown below.
*   **Environment Variable:**
    - Create a file named `.env` in the project's root directory.
    - Add the following line to your `.env` file, replacing `YOUR_API_KEY` with your actual Gemini API key:
    ```
    GEMINI_API_KEY=YOUR_API_KEY
    ```

## Installation

1.  **Clone the Repository** (If applicable, otherwise, provide other instructions)
    ```bash
    git clone https://github.com/thanhlichqnuu/laptop-advisor-chatbot-api.git
    cd laptop-advisor-chatbot-api-main
    ```

2.  **Install Dependencies:**
    It's recommended to create a virtual environment before installing dependencies:
    Install the required packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

To start the backend API, run the following command in your terminal:

```bash
uvicorn main:app --reload
```
This will start the Uvicorn server with auto-reloading enabled. The API will be accessible at http://localhost:8000/docs.

## Contact
If you have any questions or suggestions, feel free to contact us at thanhlich2103gg@gmail.com.
