"use client";

interface Stat {
  label: string;
  value: string;
  icon: React.ComponentType<{ className?: string }>;
}

interface StatsCardsProps {
  stats: Stat[];
}

export function StatsCards({ stats }: StatsCardsProps) {
  return (
    <div data-testid="stats-cards" className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      {stats.map(({ label, value, icon: Icon }) => (
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
  );
}
