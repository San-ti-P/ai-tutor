const INTENT = {
  INGEST: "ingest",
  GENERATE_EXAM: "generate_exam",
  GENERATE_EXERCISE: "generate_exercise",
  EVALUATE: "evaluate",
  QUERY_PROFILE: "query_profile",
  GENERAL_CHAT: "general_chat",
  COMPOSITE: "composite",
} as const;

type Intent = (typeof INTENT)[keyof typeof INTENT];

const MESSAGE_ROLE = {
  USER: "user",
  ASSISTANT: "assistant",
} as const;

type MessageRole = (typeof MESSAGE_ROLE)[keyof typeof MESSAGE_ROLE];

const QUESTION_TYPE = {
  MCQ: "mcq",
  OPEN: "open",
} as const;

type QuestionType = (typeof QUESTION_TYPE)[keyof typeof QUESTION_TYPE];

const DIFFICULTY = {
  EASY: "easy",
  MEDIUM: "medium",
  HARD: "hard",
} as const;

type Difficulty = (typeof DIFFICULTY)[keyof typeof DIFFICULTY];

interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
  traceId?: string;
}

interface ExamQuestion {
  id: string;
  type: QuestionType;
  prompt: string;
  options?: string[];
  baseAnswer?: string;
  sourceChunkIds?: string[];
}

interface Exam {
  id: string;
  questions: ExamQuestion[];
  topic: string;
  difficulty: Difficulty;
}

interface EvaluationResult {
  questionId: string;
  score: number;
  justification: string;
  conceptualErrors: string[];
  suggestions: string[];
}

interface ExamPreferences {
  questionTypes: QuestionType[];
  difficulty: Difficulty;
  questionCount: number;
  includeTopics: string[];
  excludeTopics: string[];
}

interface StudentProfile {
  id: string;
  topicScores: Record<string, number[]>;
  weakTopics: string[];
  preferences: ExamPreferences;
  sessionCount: number;
}

interface IngestResult {
  status: string;
  classification: string;
  topicsDetected: string[];
  chunksCreated: number;
  classificationConfidence?: number;
  lowConfidenceOcr?: { expression: string; confidence: number }[];
  documentId?: string;
}

interface ApiResponse<T> {
  data: T;
  error?: string;
}

interface ChatRequest {
  session_id: string;
  message: string;
}

interface ChatResponse {
  response: string;
  intent: Intent;
  trace_id?: string;
}

interface ExamRequest {
  session_id: string;
  topic: string;
  preferences: ExamPreferences;
}

interface EvaluationRequest {
  session_id: string;
  exam_id: string;
  answers: Record<string, string>;
}

export { INTENT, MESSAGE_ROLE, QUESTION_TYPE, DIFFICULTY };

export type {
  Intent,
  MessageRole,
  QuestionType,
  Difficulty,
  ChatMessage,
  ExamQuestion,
  Exam,
  EvaluationResult,
  ExamPreferences,
  StudentProfile,
  IngestResult,
  ApiResponse,
  ChatRequest,
  ChatResponse,
  ExamRequest,
  EvaluationRequest,
};
