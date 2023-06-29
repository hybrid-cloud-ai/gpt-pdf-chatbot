import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

// const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

// Chat History:
// {chat_history}
// Follow Up Input: {question}
// Standalone question:`;

// const QA_PROMPT = `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
// If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
// If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

// {context}

// Question: {question}
// Helpful answer in markdown:`;

const CONDENSE_PROMPT = `次の会話とフォローアップの質問がある場合、フォローアップの質問を独立した質問に言い換えなさい。

チャット履歴:
{chat_history}
フォローアップ入力: {question}
スタンドアローンの質問:`;

const QA_PROMPT = `貴方はAIアシスタントです。以下の文脈を利用して、最後の質問に答えてください。
答えがわからない場合は、わからないと答えてください。答えを作ろうとしないでください。
質問が文脈と関係ない場合は、文脈に関係する質問にしか答えられないように調整されていることを丁重に回答してください。

{context}

質問: {question}
マークダウンで役立つ答え:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
