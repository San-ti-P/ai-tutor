"use client";

import { useState } from "react";
import { ChevronRight, ChevronDown, BookOpen } from "lucide-react";

interface TopicTreeProps {
  tree: Record<string, unknown> | null | undefined;
  /** Max depth to render (default: 3) */
  maxDepth?: number;
}

interface TreeNodeState {
  expanded: boolean;
}

function TreeNode({
  name,
  children,
  depth,
  maxDepth,
}: {
  name: string;
  children: Record<string, unknown> | undefined;
  depth: number;
  maxDepth: number;
}) {
  const [expanded, setExpanded] = useState(depth === 0);
  const hasChildren =
    children &&
    typeof children === "object" &&
    Object.keys(children).length > 0;
  const isLeaf = !hasChildren || depth >= maxDepth;

  return (
    <li>
      <button
        onClick={() => hasChildren && setExpanded(!expanded)}
        disabled={isLeaf}
        className={`flex items-center gap-1.5 rounded px-1.5 py-1 text-left text-sm transition-colors hover:bg-muted/50 disabled:cursor-default ${
          depth === 0
            ? "font-semibold text-foreground"
            : "text-muted-foreground"
        }`}
      >
        {isLeaf ? (
          <span className="size-4 shrink-0" />
        ) : expanded ? (
          <ChevronDown className="size-3.5 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" />
        )}
        {isLeaf && depth > 0 && (
          <BookOpen className="size-3 shrink-0 text-muted-foreground/50" />
        )}
        <span className="truncate">{name}</span>
        {hasChildren && !isLeaf && (
          <span className="ml-auto text-muted-foreground/50 text-xs tabular-nums">
            {Object.keys(children!).length}
          </span>
        )}
      </button>

      {hasChildren && expanded && !isLeaf && (
        <ul className="ml-4 border-muted/30 border-l pl-2">
          {Object.entries(children!).map(([childName, childValue]) => (
            <TreeNode
              key={childName}
              name={childName}
              children={
                childValue && typeof childValue === "object"
                  ? (childValue as Record<string, unknown>)
                  : undefined
              }
              depth={depth + 1}
              maxDepth={maxDepth}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

export function TopicTree({ tree, maxDepth = 3 }: TopicTreeProps) {
  if (!tree || typeof tree !== "object" || Object.keys(tree).length === 0) {
    return null;
  }

  const roots = Object.entries(tree);

  return (
    <div data-testid="topic-tree" className="rounded-lg border bg-card p-4">
      <h3 className="mb-3 font-semibold text-foreground text-sm">
        Índice Temático
      </h3>
      <ul className="space-y-0.5">
        {roots.map(([name, value]) => (
          <TreeNode
            key={name}
            name={name}
            children={
              value && typeof value === "object"
                ? (value as Record<string, unknown>)
                : undefined
            }
            depth={0}
            maxDepth={maxDepth}
          />
        ))}
      </ul>
    </div>
  );
}
