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
  Session,
  SessionFile,
  SessionProfile,
  SessionCreate,
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

async function apiUpload<T>(
  path: string,
  files: File[],
  sessionId?: string,
): Promise<T> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  if (sessionId) {
    formData.append("session_id", sessionId);
  }

  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    throw new Error(`UPLOAD ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

async function apiDelete<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { method: "DELETE" });
  if (!res.ok) {
    throw new Error(`DELETE ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

async function apiPatch<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`PATCH ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  chat: (req: ChatRequest) =>
    apiPost<ApiResponse<ChatResponse>>("/api/chat", req),

  uploadDocuments: (files: File[], sessionId?: string) =>
    apiUpload<ApiResponse<IngestResult[]>>("/api/ingest", files, sessionId),

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

  updatePreferences: (studentId: string, prefs: ExamRequest["preferences"]) =>
    apiPut<ApiResponse<{ status: string }>>(
      `/api/profile/${studentId}/preferences`,
      prefs,
    ),

  // ── Session lifecycle ────────────────────────────────────────────────

  listSessions: (studentId: string) =>
    apiGet<ApiResponse<Session[]>>(
      `/api/sessions?student_id=${encodeURIComponent(studentId)}`,
    ),

  createSession: (req: SessionCreate) =>
    apiPost<ApiResponse<Session>>("/api/sessions", req),

  getSession: (sessionId: string) =>
    apiGet<ApiResponse<Session>>(`/api/sessions/${sessionId}`),

  deleteSession: (sessionId: string) =>
    apiDelete<ApiResponse<{ deleted: string }>>(`/api/sessions/${sessionId}`),

  renameSession: (sessionId: string, name: string, description?: string) =>
    apiPatch<ApiResponse<Session>>(`/api/sessions/${sessionId}`, {
      name,
      description,
    }),

  getSessionFiles: (sessionId: string) =>
    apiGet<ApiResponse<SessionFile[]>>(`/api/sessions/${sessionId}/files`),

  getSessionProfile: (sessionId: string) =>
    apiGet<ApiResponse<SessionProfile>>(`/api/sessions/${sessionId}/profile`),
};
