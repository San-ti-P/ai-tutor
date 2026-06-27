"use client";

import { useState, useEffect, useCallback } from "react";
import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { TopicTree } from "./TopicTree";
import type { SessionFile } from "@/lib/types";

interface SessionFileListProps {
  sessionId: string;
}

function classificationBadge(cls: string): {
  text: string;
  variant: "default" | "success" | "warning";
} {
  switch (cls) {
    case "apunte":
      return { text: "Apunte", variant: "default" };
    case "examen":
      return { text: "Examen", variant: "warning" };
    case "solucion":
      return { text: "Solución", variant: "success" };
    default:
      return { text: cls || "Documento", variant: "default" };
  }
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString("es-AR", {
      day: "2-digit",
      month: "short",
      year: "numeric",
    });
  } catch {
    return iso;
  }
}

export function SessionFileList({ sessionId }: SessionFileListProps) {
  const [files, setFiles] = useState<SessionFile[]>([]);
  const [loading, setLoading] = useState(false);

  const loadFiles = useCallback(async () => {
    if (!sessionId) {
      setFiles([]);
      return;
    }
    setLoading(true);
    try {
      const res = await api.getSessionFiles(sessionId);
      setFiles(res.data);
    } catch {
      setFiles([]);
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    loadFiles();
  }, [loadFiles]);

  if (!sessionId) return null;

  if (loading && files.length === 0) {
    return (
      <div className="flex flex-col gap-2 p-3">
        <p className="text-muted-foreground text-xs">Cargando archivos...</p>
      </div>
    );
  }

  if (files.length === 0) {
    return (
      <div className="flex flex-col gap-2 rounded-md border border-dashed border-border p-4 text-center">
        <p className="text-muted-foreground text-sm">
          Sin archivos en esta sesión
        </p>
        <p className="text-muted-foreground text-xs">
          Arrastrá archivos al dropzone para subir apuntes, exámenes o
          soluciones.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <h3 className="font-medium text-sm text-foreground">
        Archivos ({files.length})
      </h3>
      {files.map((f) => {
        const badge = classificationBadge(f.classification);
        return (
          <div
            key={f.id}
            className="flex flex-col gap-2 rounded-md border border-border bg-background p-3 text-sm"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="truncate font-medium text-foreground">
                {f.fileName}
              </span>
              <Badge variant={badge.variant}>{badge.text}</Badge>
            </div>

            {f.topicTree ? (
              <TopicTree tree={f.topicTree} maxDepth={3} />
            ) : (
              <div className="flex flex-wrap items-center gap-1">
                {f.topics.slice(0, 8).map((t) => (
                  <Badge key={t} variant="default">
                    {t}
                  </Badge>
                ))}
                {f.topics.length > 8 && (
                  <span className="text-muted-foreground text-xs">
                    +{f.topics.length - 8} más
                  </span>
                )}
              </div>
            )}

            <div className="flex items-center gap-3 text-muted-foreground text-xs">
              <span>{f.chunksCount} chunks</span>
              <span>{formatDate(f.ingestedAt)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
