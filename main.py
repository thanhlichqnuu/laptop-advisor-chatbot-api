from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
df = pd.read_csv('products.csv')
llm = ChatGoogleGenerativeAI(
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GEMINI_KEY,
    verbose=True,
    model="gemini-1.5-flash",
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """
         ###Instruction### Bạn là một chatbot chuyên hỗ trợ người dùng tìm kiếm thông tin về sản phẩm laptop. 
         Bạn chỉ trả lời các câu hỏi liên quan đến 5 kịch bản sau:

         1. **Gợi ý sản phẩm theo khoảng giá:**
            - Khi người dùng hỏi: "Các sản phẩm laptop có range giá từ X đến Y VND."
            - Bạn cần trả lời bằng cách liệt kê các sản phẩm có giá nằm trong khoảng từ X đến Y VND.
            - Nếu không có sản phẩm nào phù hợp, hãy trả lời: "Không tìm thấy sản phẩm nào trong khoảng giá từ X đến Y VND."
            - Ví dụ: "Dưới đây là các sản phẩm laptop trong khoảng giá từ 15,000,000 đến 20,000,000 VND: [Liệt kê sản phẩm]."

         2. **Liệt kê sản phẩm theo hãng:**
            - Khi người dùng hỏi: "Liệt kê các sản phẩm laptop thuộc hãng Z."
            - Bạn cần trả lời bằng cách liệt kê tất cả các sản phẩm thuộc hãng Z.
            - Nếu không có sản phẩm nào thuộc hãng Z, hãy trả lời: "Không tìm thấy sản phẩm nào thuộc hãng Z."
            - Ví dụ: "Dưới đây là các sản phẩm thuộc hãng ASUS: [Liệt kê sản phẩm]."

         3. **Tìm sản phẩm có giá rẻ nhất:**
            - Khi người dùng hỏi: "Sản phẩm laptop có giá rẻ hoặc thấp nhất là?"
            - Bạn cần trả lời bằng cách tìm và hiển thị sản phẩm có giá thấp nhất trong danh sách.
            - Nếu không có sản phẩm nào, hãy trả lời: "Không tìm thấy sản phẩm nào trong danh sách."
            - Ví dụ: "Sản phẩm có giá rẻ nhất là: [Tên sản phẩm] - [Giá] VND."

         4. **Tìm sản phẩm có giá lớn nhất:**
            - Khi người dùng hỏi: "Sản phẩm laptop có giá đắt hoặc lớn nhất là?"
            - Bạn cần trả lời bằng cách tìm và hiển thị sản phẩm có giá cao nhất trong danh sách.
            - Nếu không có sản phẩm nào, hãy trả lời: "Không tìm thấy sản phẩm nào trong danh sách."
            - Ví dụ: "Sản phẩm có giá đắt nhất là: [Tên sản phẩm] - [Giá] VND."
            
         5. **Xem thông tin chi tiết của sản phẩm:**
            - Khi người dùng hỏi: "Xem thông tin chi tiết của sản phẩm X."
            - Bạn cần trả lời bằng cách hiển thị thông tin chi tiết của sản phẩm X.
            - Nếu không tìm thấy sản phẩm, hãy trả lời: "Không tìm thấy sản phẩm có tên X."
            - Ví dụ: "Thông tin chi tiết của sản phẩm X: [Thông tin chi tiết]."

         ###Rules###
         - Chỉ trả lời các câu hỏi liên quan đến 5 kịch bản trên. Nếu người dùng hỏi ngoài phạm vi này, hãy trả lời: "Xin lỗi, tôi chỉ hỗ trợ các câu hỏi liên quan đến tư vấn sản phẩm laptop."
         - Luôn trả lời một cách rõ ràng, chuyên nghiệp và thân thiện.
         - Sử dụng ngôn ngữ tự nhiên, gần gũi với người dùng.

         Context:
         {context}
         """),
        ("placeholder", "{history}"),
        ("human", "{input}")
    ]
)

chain = prompt_template | llm

import pandas as pd

df = pd.read_csv("products.csv")

def call_chat(query, history):
    context = ""
    
    # 1. Gợi ý sản phẩm theo khoảng giá
    if "range giá từ" in query.lower():
        price_range = query.lower().split("range giá từ")[1].strip().split("đến")
        min_price = float(price_range[0].strip().replace(",", "").replace(" vnd", ""))
        max_price = float(price_range[1].strip().replace(",", "").replace(" vnd", ""))
        products = df[(df['price'] >= min_price) & (df['price'] <= max_price)]
        if not products.empty:
            context = "\n".join([f"{row['name']} - {row['price']} VND" for _, row in products.iterrows()])
        else:
            context = f"Không tìm thấy sản phẩm nào trong khoảng giá từ {min_price} đến {max_price} VND."

    # 2. Liệt kê sản phẩm theo hãng
    elif "thuộc hãng" in query.lower():
        brand = query.lower().split("thuộc hãng")[1].strip()
        products = df[df['factory'].str.contains(brand, case=False)]
        if not products.empty:
            context = "\n".join([f"{row['name']} - {row['price']} VND" for _, row in products.iterrows()])
        else:
            context = f"Không tìm thấy sản phẩm nào thuộc hãng {brand}."

    # 3. Tìm sản phẩm có giá rẻ nhất
    elif "giá rẻ nhất" in query.lower():
        cheapest = df.loc[df['price'].idxmin()]
        context = f"Sản phẩm có giá rẻ nhất là: {cheapest['name']} - {cheapest['price']} VND"

    # 4. Tìm sản phẩm có giá lớn nhất
    elif "giá lớn nhất" in query.lower() or "giá cao nhất" in query.lower():
        most_expensive = df.loc[df['price'].idxmax()]
        context = f"Sản phẩm có giá lớn nhất là: {most_expensive['name']} - {most_expensive['price']} VND"

    # 5. Tìm thông tin chi tiết của sản phẩm
    elif "thông tin chi tiết của sản phẩm" in query.lower():
        product_name = query.lower().split("thông tin chi tiết của sản phẩm")[1].strip()
        product = df[df['name'].str.contains(product_name, case=False)]
        if not product.empty:
            product_info = product.iloc[0]
            context = f"Thông tin chi tiết của sản phẩm {product_info['name']}:\n" \
                      f"- Hãng: {product_info['factory']}\n" \
                      f"- Giá: {product_info['price']} VND\n" \
                      f"- Mô tả ngắn: {product_info['short_desc']}\n" \
                      f"- Mô tả chi tiết: {product_info['detail_desc']}\n" \
                      f"- Số lượng còn lại: {product_info['quantity']}\n" \
                      f"- Đối tượng sử dụng: {product_info['target']}"

    else:
        context = "Xin lỗi, tôi chỉ hỗ trợ các câu hỏi liên quan đến gợi ý sản phẩm theo khoảng giá, liệt kê sản phẩm theo hãng, tìm sản phẩm có giá rẻ nhất, tìm sản phẩm có giá lớn nhất, tìm sản phẩm đắt nhất của hãng, tìm sản phẩm rẻ nhất của hãng, liệt kê sản phẩm có chip cụ thể, tìm sản phẩm có chip cụ thể rẻ nhất, và tìm sản phẩm có chip cụ thể đắt nhất."

    history.append({"role": "user", "content": query})
    response = chain.invoke({"context": context, "input": query, "history": history})
    history.append({"role": "assistant", "content": response.content})

    return response.content, history

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  
)

class ChatRequest(BaseModel):
    query: str
    history: list

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response, updated_history = call_chat(request.query, request.history)
        return {"result": response, "history": updated_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)