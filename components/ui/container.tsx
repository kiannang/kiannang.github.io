import { cn } from "@/lib/utils";

type ContainerProps = {
  children: React.ReactNode;
  className?: string;
  id?: string;
};

export function Container({ children, className, id }: ContainerProps) {
  return (
    <section id={id} className={cn("scroll-mt-28 py-8", className)}>
      {children}
    </section>
  );
}
