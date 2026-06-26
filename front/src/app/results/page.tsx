import { Suspense } from "react";
import { ResultsContent } from "./ResultsContent";

function ResultsLoading() {
  return (
    <div className="flex flex-col items-center gap-4 py-16">
      <div className="size-8 animate-spin rounded-full border-2 border-muted-foreground/20 border-t-muted-foreground" />
      <p className="text-muted-foreground text-sm">Cargando resultados...</p>
    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={<ResultsLoading />}>
      <ResultsContent />
    </Suspense>
  );
}
