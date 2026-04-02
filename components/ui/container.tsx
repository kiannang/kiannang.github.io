import { cn } from "@/lib/utils";

type ContainerProps = {
  children: React.ReactNode;
  className?: string;
  id?: string;
};

export function Container({ children, className, id }: ContainerProps) {
  return (
    <section
      id={id}
      className={cn(
        "relative overflow-hidden rounded-3xl border border-line bg-panel shadow-glow backdrop-blur",
        className,
      )}
    >
      {children}
    </section>
  );
}
