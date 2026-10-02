Multimodal AI Customer Support Agent

**A Multimodal AI Customer Support Agent for Power Bank After-Sales Service**


This project is a multimodal AI customer support agent designed for power bank after-sales service. It combines multi-turn conversations, image understanding, official knowledge retrieval, historical case search, and business rules to assist with product identification, fault diagnosis, safety assessment, troubleshooting, warranty enquiries, and human-agent escalation.

This project was independently developed by a four-person team for the Anker 1st Hackathon Challenge. It is a participant project and is not an official Anker product.

## Team

### Zhang Xiaoyu — Team Lead and Lead AI/Backend Engineer

**Email:** 928398910@qq.com

Led the project from initial concept to final delivery and took primary responsibility for the system architecture and core implementation.

- Led requirements analysis, solution design, technology selection, and development planning.
- Designed and implemented the core backend services.
- Built the LangGraph agent workflow, routing logic, and case-handling process.
- Developed the RAG pipeline for knowledge-base and historical-case retrieval.
- Implemented SQLite-based management for cases, conversations, images, and diagnostic results.
- Integrated and refined the Doubao multimodal models for customer conversations, semantic retrieval, and image analysis.
- Coordinated frontend-backend integration and led end-to-end testing and final delivery.

### Zhang Yichi — User Experience and Quality Assurance

**Email:** yzhangwb@connect.ust.hk

- Analysed the end-to-end business and customer-support workflow.
- Tested core user journeys and system responses.
- Helped improve customer interactions and after-sales handling processes.

### Qin Yuting — Product Strategy and Presentation

**Email:** 1678754058@qq.com

- Structured and refined the overall project proposal.
- Organised the technical architecture and solution narrative.
- Produced the presentation deck and competition materials.

### Ren Yuexi — Frontend and Multimodal AI Engineer

**Email:** 3068692995@qq.com

- Developed the frontend interface, chat window, image-upload workflow, and case-status display.
- Improved the integration and performance of the multimodal model.
- Collaborated on frontend-backend integration and user-experience refinement.

## Key Features

- Multi-turn customer support conversations and case-status management
- Power bank model, fault, and customer-emotion identification
- Image analysis for product appearance, model labels, ports, and indicator lights
- Safety-risk detection for swelling, leakage, burn marks, deformation, and abnormal heating
- Retrieval from official troubleshooting resources and the local knowledge base
- Semantic search across similar historical support cases
- Simulated order, warranty, serial-number, and dealer-order enquiries
- Response adaptation based on detected customer emotion
- Automatic suspension of DIY troubleshooting and human escalation in high-risk scenarios
- Automatic transition of completed cases to the `resolved` state
- SQLite storage for cases, conversations, images, and diagnostic results

## Model Responsibilities

The system uses two Doubao models with separate responsibilities:

```env
# Customer conversations, emotion understanding, and image analysis
DOUBAO_MODEL=your-doubao-seed-2.1-pro-endpoint-id

# Vector retrieval for the knowledge base and historical cases
DOUBAO_EMBEDDING_MODEL=your-doubao-embedding-vision-endpoint-id
```

- `DOUBAO_MODEL` generates customer-support responses and analyses uploaded images.
- `DOUBAO_EMBEDDING_MODEL` generates embeddings for semantic retrieval.

## Project Structure

```text
multimodal-ai-customer-support-agent/
├── README.md
├── README.zh-CN.md
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── work_orders.json
│   └── knowledge_base/
│       └── powerbank_kb.json
└── frontend/
    ├── index.html
    ├── styles.css
    └── app.js
```

The backend creates the following directories on first launch:

```text
backend/data/
backend/uploads/
```

- `data/` stores the SQLite database and vector cache.
- `uploads/` stores images uploaded during local use.

Both directories should remain excluded from version control.

## Local Setup

### 1. Open the backend directory

```powershell
cd backend
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Create the local environment file

```powershell
Copy-Item .env.example .env
```

Configure the following values in `.env`:

```env
DOUBAO_API_KEY=your-api-key
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_MODEL=your-doubao-seed-2.1-pro-endpoint-id
DOUBAO_EMBEDDING_MODEL=your-doubao-embedding-vision-endpoint-id
DOUBAO_ENABLED=true
```

Never commit `.env` or any real API key to the repository.

### 4. Start the application

```powershell
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

The FastAPI backend serves the API, frontend application, and uploaded images. A separate frontend server is not required.

After startup, open:

- Application: <http://127.0.0.1:8000/app/>
- API documentation: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

## Suggested Demo Flow

1. Open `/app/` and create a new support case.
2. Enter: `My power bank is not charging.`
3. Upload product and fault images when prompted.
4. Review the detected product model and visible safety risks.
5. Continue with: `I have uploaded the images. Please continue troubleshooting.`
6. Review the retrieved knowledge-base content and similar historical cases.
7. Test standard troubleshooting or a warranty enquiry.
8. Enter: `The power bank is swollen and has a strange smell.` to verify high-risk escalation.
9. Enter: `It is charging normally now. The issue has been resolved.` to verify that the case enters the `resolved` state.

## Data and Demonstration Scope

`work_orders.json` contains fictional and anonymised demonstration cases. It does not contain real customer data.

Order, warranty, serial-number, and dealer-order enquiries use simulated business data to demonstrate agent tool use and workflow orchestration.

## Emotion Detection

The current version uses a demonstration-oriented approach that prioritises deterministic rules and supplements them with model-assisted understanding. It detects expressions associated with complaints, urgency, anxiety, disruption, and safety concerns, then adjusts the response tone and escalation strategy.

Because the project does not use a dedicated sentiment-classification model, it may misinterpret sarcasm, implicit emotions, or complex context. Future versions could combine a specialised classifier with conversation history, peak emotion levels, and repeated-complaint signals.

## Security Notes

The public repository should contain:

- No real API keys or access tokens
- No `.env` file
- No generated SQLite database
- No user-uploaded images
- No real customer, order, warranty, or dealer data

Only `.env.example` should be committed as the environment-variable template.

## Optional Public Demo

After starting the application locally, port `8000` can be exposed temporarily with Cloudflare Tunnel:

```powershell
cloudflared tunnel --url http://localhost:8000
```

The command returns a temporary public URL similar to:

```text
https://xxxxx.trycloudflare.com
```

Visitors can access the application at:

```text
https://xxxxx.trycloudflare.com/app/
```

The FastAPI process and tunnel process must remain active during the demonstration. Do not expose a development instance containing secrets or personal data.


## Disclaimer

This is an independent hackathon prototype created for demonstration and educational purposes. It is not affiliated with, endorsed by, or released as an official product of Anker Innovations.
