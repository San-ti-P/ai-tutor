import { cn } from "@/lib/utils";

interface BadgeProps {
  variant?: "default" | "success" | "warning" | "error";
  children: React.ReactNode;
  className?: string;
}

export function Badge({
  variant = "default",
  children,
  className,
}: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium",
        variant === "default" && "bg-muted text-muted-foreground",
        variant === "success" &&
          "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
        variant === "warning" &&
          "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200",
        variant === "error" &&
          "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
        className,
      )}
    >
      {children}
    </span>
  );
}
