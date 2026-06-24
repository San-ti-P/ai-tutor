import type {
  ApiResponse,
  ChatRequest,
  ChatResponse,
  ExamRequest,
  Exam,
  ExerciseRequest,
  Exercise,
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

async function apiPut<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`PUT ${path} failed: ${res.status}`);
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
    apiPost<ApiResponse<ChatResponse>>("/api/chat", req),

  uploadDocuments: (files: File[]) =>
    apiUpload<ApiResponse<IngestResult[]>>("/api/ingest", files),

  generateExam: (req: ExamRequest) =>
    apiPost<ApiResponse<Exam>>("/api/exam/generate", req),

  generateExercise: (req: ExerciseRequest) =>
    apiPost<ApiResponse<Exercise>>("/api/exercise/generate", req),

  submitAnswers: (req: EvaluationRequest) =>
    apiPost<ApiResponse<EvaluationResult[]>>("/api/evaluate", req),

  getProfile: (sessionId: string) =>
    apiGet<ApiResponse<StudentProfile>>(`/api/profile/${sessionId}`),

  getDashboard: (studentId: string) =>
    apiGet<ApiResponse<StudentProfile>>(`/api/students/${studentId}/dashboard`),

  updatePreferences: (
    studentId: string,
    prefs: ExamRequest["preferences"]
  ) =>
    apiPut<ApiResponse<{ status: string }>>(
      `/api/profile/${studentId}/preferences`,
      prefs
    ),
};
