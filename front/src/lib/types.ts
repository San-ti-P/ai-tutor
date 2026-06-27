const INTENT = {
  INGEST: "ingest",
  RETRIEVE: "retrieve",
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
  exam?: Exam;
  isError?: boolean;
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
  status?: string;
  warnings?: string[];
  topicNotFound?: string[];
  topicSuggestions?: string[];
}

interface EvaluationResult {
  questionId: string;
  score: number;
  justification: string;
  conceptualErrors: string[];
  suggestions: string[];
  isEvaluable?: boolean;
  nonEvaluableReason?: string;
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
  sessionHistory: Array<{
    id: string;
    started_at: string;
    ended_at: string | null;
    intent: string | null;
    status: string;
    questions_answered: number;
    average_score: number | null;
  }>;
}

interface IngestResult {
  sessionId: string;
  status: string;
  classification: string;
  topicsDetected: string[];
  topicTree?: Record<string, Record<string, unknown>> | null;
  chunksCreated: number;
  classificationConfidence?: number;
  lowConfidenceOcr?: { expression: string; confidence: number }[];
  documentId?: string;
}

interface ExerciseStep {
  stepNumber: number;
  description: string;
  result: string;
  sourceChunkIds: string[];
}

interface Exercise {
  exercise_id: string;
  statement: string;
  given_data?: string;
  question: string;
  model_solution: {
    steps: ExerciseStep[];
    finalAnswer: string;
    keyConcepts: string[];
    sourceChunkIds?: string[];
  };
  topics_covered: string[];
  source_chunk_ids?: string[];
  topic_not_found: string[];
  topic_suggestions: string[];
  status: string;
}

interface ExerciseRequest {
  session_id: string;
  topic: string;
  difficulty: Difficulty;
  exercise_type: string;
}

interface ApiResponse<T> {
  data: T;
  error?: string;
  trace_id?: string;
}

interface ChatRequest {
  session_id: string;
  message: string;
}

interface ChatResponse {
  response: string;
  intent: Intent;
  trace_id?: string;
  exam?: Exam;
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
  examQuestions?: ExamQuestion[];
}

interface Session {
  id: string;
  name: string;
  description: string;
  studentId: string;
  createdAt: string;
  status: string;
  fileCount: number;
  examCount: number;
  averageScore: number | null;
}

interface SessionFile {
  id: string;
  fileName: string;
  classification: string;
  topics: string[];
  topicTree?: Record<string, Record<string, unknown>> | null;
  chunksCount: number;
  ingestedAt: string;
}

interface SessionProfile {
  sessionId: string;
  topicScores: Record<string, number[]>;
  weakTopics: string[];
  examCount: number;
  averageScore: number | null;
}

interface SessionCreate {
  name: string;
  description?: string;
  studentId: string;
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
  Exercise,
  ExerciseStep,
  ExerciseRequest,
  EvaluationResult,
  ExamPreferences,
  StudentProfile,
  IngestResult,
  ApiResponse,
  ChatRequest,
  ChatResponse,
  ExamRequest,
  EvaluationRequest,
  Session,
  SessionFile,
  SessionProfile,
  SessionCreate,
};
