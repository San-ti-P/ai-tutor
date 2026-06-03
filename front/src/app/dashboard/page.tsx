import {
  BookOpen,
  TrendingUp,
  Target,
  CheckCircle2,
} from "lucide-react";

const STATS = [
  {
    label: "Sesiones completadas",
    value: "0",
    icon: BookOpen,
  },
  {
    label: "Temas dominados",
    value: "0",
    icon: Target,
  },
  {
    label: "Promedio general",
    value: "--%",
    icon: TrendingUp,
  },
  {
    label: "Ejercicios resueltos",
    value: "0",
    icon: CheckCircle2,
  },
] as const;

export default function DashboardPage() {
  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="font-bold text-3xl text-foreground tracking-tight">
          Mi Progreso
        </h1>
        <p className="mt-1 text-muted-foreground">
          Seguimiento de tu rendimiento académico
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {STATS.map(({ label, value, icon: Icon }) => (
          <div
            key={label}
            className="flex flex-col gap-2 rounded-lg border border-border bg-card p-4"
          >
            <Icon className="size-5 text-muted-foreground" />
            <div>
              <p className="font-bold text-2xl text-foreground">{value}</p>
              <p className="text-muted-foreground text-sm">{label}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="rounded-lg border border-border bg-card p-8 text-center">
        <p className="text-muted-foreground text-sm">
          Próximamente: gráficos de evolución por tema
        </p>
      </div>
    </div>
  );
}
