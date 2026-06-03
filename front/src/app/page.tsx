"use client";

import { Send } from "lucide-react";

export default function ChatPage() {
  return (
    <div className="flex flex-col gap-6">
      <div className="text-center">
        <h1 className="font-bold text-3xl text-foreground tracking-tight">
          Tutor Académico
        </h1>
        <p className="mt-2 text-muted-foreground">
          Prepará tus exámenes con IA
        </p>
      </div>

      <div className="flex flex-1 flex-col rounded-lg border border-border bg-card">
        <div className="flex flex-1 items-center justify-center p-12">
          <p className="text-center text-muted-foreground text-sm">
            Todavía no hay mensajes. Escribí tu consulta abajo.
          </p>
        </div>

        <div className="border-t border-border p-4">
          <div className="flex gap-2">
            <input
              type="text"
              disabled
              placeholder="El chat estará disponible pronto..."
              className="flex-1 rounded-md border border-border bg-muted px-3 py-2 text-sm text-muted-foreground placeholder:text-muted-foreground/60"
            />
            <button
              type="button"
              disabled
              className="inline-flex items-center gap-1.5 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground opacity-50 transition-opacity"
            >
              <Send className="size-4" />
              Enviar
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
