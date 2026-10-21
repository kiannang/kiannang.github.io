import { cn } from "@/lib/utils";

type PanelProps = {
  title?: React.ReactNode;
  aside?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  id?: string;
};

// A white box with an optional gradient title strip, like the 2001 apple.com sidebars.
export function Panel({ title, aside, children, className, id }: PanelProps) {
  return (
    <div id={id} className={cn("aqua-box scroll-mt-28 overflow-hidden", className)}>
      {title && (
        <div className="aqua-bar flex items-center justify-between gap-3 border-x-0 border-t-0 px-3 py-1 text-[12px]">
          <span className="font-bold text-foreground">{title}</span>
          {aside && <span className="shrink-0 text-muted">{aside}</span>}
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  );
}
