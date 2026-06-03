import type {
  ApiResponse,
  ChatRequest,
  ChatResponse,
  ExamRequest,
  Exam,
  EvaluationRequest,
  EvaluationResult,
  IngestResult,
  StudentProfile,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`);
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`POST ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

async function apiUpload<T>(path: string, files: File[]): Promise<T> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    throw new Error(`UPLOAD ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  chat: (req: ChatRequest) =>
    apiPost<ApiResponse<ChatResponse>>("/chat", req),

  uploadDocuments: (files: File[]) =>
    apiUpload<ApiResponse<IngestResult>>("/ingest", files),

  generateExam: (req: ExamRequest) =>
    apiPost<ApiResponse<Exam>>("/exam/generate", req),

  submitAnswers: (req: EvaluationRequest) =>
    apiPost<ApiResponse<EvaluationResult[]>>("/evaluate", req),

  getProfile: (sessionId: string) =>
    apiGet<ApiResponse<StudentProfile>>(`/profile/${sessionId}`),
};
