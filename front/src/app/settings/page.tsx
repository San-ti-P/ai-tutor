import { Settings } from "lucide-react";

export default function SettingsPage() {
  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="font-bold text-3xl text-foreground tracking-tight">
          Configuración
        </h1>
        <p className="mt-1 text-muted-foreground">
          Personalizá tu experiencia de estudio
        </p>
      </div>

      <div className="flex flex-col gap-6 rounded-lg border border-border bg-card p-6">
        <div className="flex flex-col gap-1.5">
          <label className="font-medium text-foreground text-sm">
            Preferencias de examen
          </label>
          <input
            type="text"
            disabled
            placeholder="Multiple choice, Respuesta libre"
            className="rounded-md border border-border bg-muted px-3 py-2 text-sm text-muted-foreground placeholder:text-muted-foreground/60"
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="font-medium text-foreground text-sm">
            Temas a priorizar
          </label>
          <input
            type="text"
            disabled
            placeholder="Álgebra, Cálculo, Probabilidad..."
            className="rounded-md border border-border bg-muted px-3 py-2 text-sm text-muted-foreground placeholder:text-muted-foreground/60"
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="font-medium text-foreground text-sm">
            Dificultad preferida
          </label>
          <select
            disabled
            className="rounded-md border border-border bg-muted px-3 py-2 text-sm text-muted-foreground"
          >
            <option>Medio</option>
          </select>
        </div>
      </div>

      <div className="flex items-center gap-2 rounded-lg border border-border bg-muted p-4 text-muted-foreground text-sm">
        <Settings className="size-4 shrink-0" />
        <p>La configuración estará disponible pronto.</p>
      </div>
    </div>
  );
}
