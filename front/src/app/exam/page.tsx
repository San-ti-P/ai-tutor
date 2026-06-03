import { FileText, Zap } from "lucide-react";

export default function ExamPage() {
  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="font-bold text-3xl text-foreground tracking-tight">
          Generar Examen
        </h1>
        <p className="mt-1 text-muted-foreground">
          Creá exámenes personalizados basados en tu material de estudio
        </p>
      </div>

      <div className="rounded-lg border border-border bg-card p-6">
        <div className="flex flex-col gap-6">
          <div className="flex flex-col gap-1.5">
            <label className="font-medium text-foreground text-sm">
              Tema del examen
            </label>
            <input
              type="text"
              disabled
              placeholder="Tema del examen"
              className="rounded-md border border-border bg-muted px-3 py-2 text-sm text-muted-foreground placeholder:text-muted-foreground/60"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="font-medium text-foreground text-sm">
              Dificultad
            </label>
            <div className="flex gap-2">
              {["Fácil", "Medio", "Difícil"].map((level) => (
                <button
                  key={level}
                  type="button"
                  disabled
                  className="rounded-md border border-border bg-muted px-4 py-1.5 text-muted-foreground text-sm"
                >
                  {level}
                </button>
              ))}
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="font-medium text-foreground text-sm">
              Cantidad de preguntas
            </label>
            <input
              type="number"
              disabled
              placeholder="10"
              className="w-24 rounded-md border border-border bg-muted px-3 py-2 text-sm text-muted-foreground"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="font-medium text-foreground text-sm">
              Tipo de preguntas
            </label>
            <div className="flex flex-col gap-1">
              {["Multiple choice", "Respuesta libre"].map((type) => (
                <label
                  key={type}
                  className="inline-flex items-center gap-2 text-muted-foreground text-sm"
                >
                  <input type="checkbox" disabled className="opacity-50" />
                  {type}
                </label>
              ))}
            </div>
          </div>

          <button
            type="button"
            disabled
            className="inline-flex items-center gap-1.5 self-start rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground opacity-50"
          >
            <Zap className="size-4" />
            Generar Examen
          </button>
        </div>
      </div>

      <div className="flex items-center gap-2 rounded-lg border border-border bg-accent p-4 text-accent-foreground text-sm">
        <FileText className="size-4 shrink-0" />
        <p>
          Cargá material de estudio primero para generar exámenes
          personalizados.
        </p>
      </div>
    </div>
  );
}
