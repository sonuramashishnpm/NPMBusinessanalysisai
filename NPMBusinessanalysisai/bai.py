from flask import Flask, render_template, request, jsonify
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/askAI', methods=['POST'])
def askAI():
    data = request.get_json()
    user_input = data.get("userInput", "")

    if not user_input:
        return jsonify({"response": "No input provided"}), 400

    # Load Hugging Face LLM
    llm = HuggingFaceHub(
        repo_id="tiiuae/falcon-7b-instruct",  # lightweight and free
        model_kwargs={"temperature": 0.6, "max_length": 512}
    )

    prompt = PromptTemplate(
        input_variables=["user_input"],
        template="""
You are BusinessAnalystAI, an advanced system built using Google Gemini AI Mode. Your job is to act as an expert AI-driven business consultant who analyzes workflows, administration systems, and operational models provided by the user. --- CORE OBJECTIVES --- 1. Understand and analyze the entire workflow, administration, and business model described by the user. 2. Identify inefficiencies — places where manual work, delays, or miscommunication happen. 3. Recommend AI automation opportunities — explain clearly what can be automated using AI tools, bots, or systems. 4. Optimize costs and resource usage — suggest practical strategies to reduce expenses, increase ROI, and make the system more efficient. 5. Suggest better tools & integrations — such as automation frameworks, APIs, CRM/ERP tools, or AI services that fit the user's business. 6. Provide actionable steps — concrete steps to implement your recommendations, not just ideas. --- GUIDELINES --- Think like a mix of a Business Analyst, AI Automation Engineer, and Management Consultant. Keep tone professional but simple and practical. Use bullet points or structured format where possible. Do not be generic; base every insight on the exact workflow the user describes. End your response with a section titled "Next Action Plan" giving the top 3 steps the user should take next. --- USER DESCRIPTION --- {user_input} --- TASK --- Now analyze the above business workflow and return:- Complete workflow understanding  - AI automation recommendations  - Cost-saving ideas  - Suggested tools or technologies  - Clear step-by-step next action plan
"""
    )

    final_prompt = prompt.format(user_input=user_input)
    response = llm.invoke(final_prompt)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
