"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface TopicChartProps {
  data: { topic: string; score: number }[];
}

function barColor(score: number): string {
  if (score >= 7) return "hsl(142, 76%, 36%)";
  if (score >= 5) return "hsl(45, 93%, 47%)";
  return "hsl(0, 84%, 60%)";
}

export function TopicChart({ data }: TopicChartProps) {
  if (data.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-8 text-center">
        <p className="text-muted-foreground text-sm">
          Sin datos de temas. Completá evaluaciones para ver tu progreso.
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-card p-6">
      <h3 className="mb-4 font-semibold text-foreground text-sm">
        Puntajes por tema
      </h3>
      <div data-testid="topic-chart">
      <ResponsiveContainer width="100%" height={250}>
        <BarChart
          data={data}
          margin={{ top: 5, right: 20, left: 0, bottom: 40 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis
            dataKey="topic"
            tick={{ fontSize: 11 }}
            angle={-30}
            textAnchor="end"
            interval={0}
            stroke="hsl(var(--muted-foreground))"
          />
          <YAxis
            domain={[0, 10]}
            tick={{ fontSize: 11 }}
            stroke="hsl(var(--muted-foreground))"
          />
          <Tooltip
            contentStyle={{
              background: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "var(--radius)",
              fontSize: "12px",
            }}
          />
          <Bar dataKey="score" radius={[4, 4, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={index} fill={barColor(entry.score)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      </div>
    </div>
  );
}
