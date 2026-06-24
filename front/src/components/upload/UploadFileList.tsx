"use client";

import { Spinner } from "@/components/ui/spinner";
import { Badge } from "@/components/ui/badge";

export type FileStatus = "pending" | "uploading" | "complete" | "error" | "rejected";

interface FileEntry {
  file: File;
  status: FileStatus;
  message?: string;
  result?: {
    classification?: string;
    topicsDetected?: string[];
    chunksCreated?: number;
  };
}

interface UploadFileListProps {
  files: FileEntry[];
}

function statusBadge(status: FileStatus): {
  text: string;
  variant: "default" | "success" | "warning" | "error";
} {
  switch (status) {
    case "uploading":
      return { text: "Subiendo...", variant: "warning" };
    case "complete":
      return { text: "Completado", variant: "success" };
    case "error":
      return { text: "Error", variant: "error" };
    case "rejected":
      return { text: "Rechazado", variant: "error" };
    default:
      return { text: "Pendiente", variant: "default" };
  }
}

export function UploadFileList({ files }: UploadFileListProps) {
  if (files.length === 0) return null;

  return (
    <div className="flex flex-col gap-2">
      {files.map((entry, i) => {
        const badge = statusBadge(entry.status);
        return (
          <div
            key={`${entry.file.name}-${i}`}
            className="flex flex-col gap-1 rounded-md border border-border bg-background p-3 text-sm"
          >
            <div className="flex items-center justify-between">
              <span className="truncate font-medium text-foreground">
                {entry.file.name}
              </span>
              <div className="flex shrink-0 items-center gap-2">
                {entry.status === "uploading" && <Spinner size="sm" />}
                <Badge variant={badge.variant}>{badge.text}</Badge>
              </div>
            </div>
            <span className="text-muted-foreground text-xs">
              {(entry.file.size / 1024).toFixed(1)} KB
            </span>
            {entry.message && (
              <p className="text-muted-foreground text-xs">{entry.message}</p>
            )}
            {entry.result && entry.status === "complete" && (
              <div className="mt-1 flex flex-wrap gap-1">
                {entry.result.topicsDetected?.map((t) => (
                  <Badge key={t} variant="default">
                    {t}
                  </Badge>
                ))}
                {entry.result.chunksCreated !== undefined && (
                  <span className="text-muted-foreground text-xs">
                    {entry.result.chunksCreated} chunks creados
                  </span>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
