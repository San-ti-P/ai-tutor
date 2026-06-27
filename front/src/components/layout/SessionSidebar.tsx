"use client";

import { useState, useCallback } from "react";
import type { Session } from "@/lib/types";

interface SessionSidebarProps {
  sessions: Session[];
  activeSession: Session | null;
  onCreate: (name: string, description?: string) => Promise<Session>;
  onRename: (id: string, name: string) => Promise<void>;
  onSwitch: (id: string) => void;
  onDelete: (id: string) => Promise<void>;
}

export function SessionSidebar({
  sessions,
  activeSession,
  onCreate,
  onRename,
  onSwitch,
  onDelete,
}: SessionSidebarProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [showCreate, setShowCreate] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [renaming, setRenaming] = useState(false);

  const handleRename = useCallback(
    async (id: string) => {
      const trimmed = editName.trim();
      if (!trimmed || trimmed === sessions.find((s) => s.id === id)?.name) {
        setEditingId(null);
        return;
      }
      setRenaming(true);
      try {
        await onRename(id, trimmed);
        setEditingId(null);
      } catch {
        // keep editing state on error
      } finally {
        setRenaming(false);
      }
    },
    [editName, onRename, sessions],
  );

  const handleDelete = useCallback(
    async (id: string) => {
      setDeleting(id);
      try {
        await onDelete(id);
      } finally {
        setDeleting(null);
      }
    },
    [onDelete],
  );

  return (
    <>
      <aside
        className={`flex flex-col border-r border-border bg-card transition-all ${
          collapsed ? "w-12" : "w-64"
        }`}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-3 border-b border-border">
          {!collapsed && (
            <span className="font-semibold text-sm text-foreground">
              Sesiones
            </span>
          )}
          <button
            type="button"
            onClick={() => setCollapsed((c) => !c)}
            className="ml-auto text-muted-foreground hover:text-foreground text-xs p-1 rounded"
            title={collapsed ? "Expandir" : "Colapsar"}
          >
            {collapsed ? "▶" : "◀"}
          </button>
        </div>

        {/* Session list */}
        {!collapsed && (
          <div className="flex flex-col flex-1 overflow-y-auto p-2 gap-1">
            {sessions.length === 0 ? (
              <p className="text-xs text-muted-foreground p-2">Sin sesiones</p>
            ) : (
              sessions.map((s) => {
                const isActive = activeSession?.id === s.id;
                const isEditing = editingId === s.id;
                return (
                  <div
                    key={s.id}
                    data-testid="session-item"
                    data-session-id={s.id}
                    className={`group flex items-center justify-between rounded-md px-2 py-1.5 text-sm transition-colors ${
                      isActive
                        ? "bg-primary/15 text-primary font-medium"
                        : "hover:bg-accent text-foreground"
                    }`}
                  >
                    {isEditing ? (
                      <input
                        type="text"
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") handleRename(s.id);
                          if (e.key === "Escape") setEditingId(null);
                        }}
                        onBlur={() => handleRename(s.id)}
                        disabled={renaming}
                        autoFocus
                        className="flex-1 bg-transparent border-b border-primary outline-none text-sm px-1 py-0 min-w-0"
                      />
                    ) : (
                      <button
                        type="button"
                        onClick={() => onSwitch(s.id)}
                        onDoubleClick={() => {
                          setEditingId(s.id);
                          setEditName(s.name);
                        }}
                        className="flex-1 text-left truncate"
                        title={`${s.name} — doble clic para renombrar`}
                      >
                        {s.name}
                      </button>
                    )}
                    <div className="flex items-center ml-1">
                      {isActive && !isEditing && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setEditingId(s.id);
                            setEditName(s.name);
                          }}
                          className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-foreground text-xs px-1"
                          title="Renombrar"
                        >
                          ✎
                        </button>
                      )}
                      {isActive && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDelete(s.id);
                          }}
                          disabled={deleting === s.id}
                          className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive text-xs px-1"
                          title="Eliminar sesión"
                        >
                          {deleting === s.id ? "…" : "✕"}
                        </button>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        )}

        {/* Create button */}
        {!collapsed && (
          <div className="p-2 border-t border-border">
            <button
              type="button"
              onClick={() => setShowCreate(true)}
              data-testid="new-session-btn"
              className="w-full rounded-md bg-primary text-primary-foreground text-sm py-1.5 hover:bg-primary/90 transition-colors"
            >
              + Nueva sesión
            </button>
          </div>
        )}
      </aside>

      {showCreate && (
        <SessionCreateModal
          onSubmit={async (name, desc) => {
            await onCreate(name, desc);
            setShowCreate(false);
          }}
          onClose={() => setShowCreate(false)}
        />
      )}
    </>
  );
}

function SessionCreateModal({
  onSubmit,
  onClose,
}: {
  onSubmit: (name: string, description?: string) => Promise<void>;
  onClose: () => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = useCallback(async () => {
    const trimmed = name.trim();
    if (!trimmed) {
      setError("El nombre es obligatorio");
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      await onSubmit(trimmed, description.trim() || undefined);
    } catch {
      setError("No se pudo crear la sesión");
    } finally {
      setSubmitting(false);
    }
  }, [name, description, onSubmit]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-card rounded-lg border border-border p-6 w-full max-w-sm shadow-xl">
        <h2 className="font-semibold text-lg text-foreground mb-4">
          Nueva sesión
        </h2>

        <div className="flex flex-col gap-3">
          <div>
            <label className="block text-sm font-medium text-foreground mb-1">
              Nombre <span className="text-destructive">*</span>
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Ej: Cálculo I"
              data-testid="session-name-input"
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              autoFocus
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSubmit();
                if (e.key === "Escape") onClose();
              }}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-foreground mb-1">
              Descripción
            </label>
            <input
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Opcional"
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSubmit();
                if (e.key === "Escape") onClose();
              }}
            />
          </div>

          {error && <p className="text-xs text-destructive">{error}</p>}

          <div className="flex gap-2 justify-end mt-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-md border border-input bg-background px-4 py-2 text-sm text-foreground hover:bg-accent"
              disabled={submitting}
            >
              Cancelar
            </button>
            <button
              type="button"
              onClick={handleSubmit}
              disabled={submitting || !name.trim()}
              data-testid="session-create-confirm"
              className="rounded-md bg-primary text-primary-foreground px-4 py-2 text-sm hover:bg-primary/90 disabled:opacity-50"
            >
              {submitting ? "Creando..." : "Crear"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
