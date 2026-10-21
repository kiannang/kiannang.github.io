import Link from "next/link";

type SectionHeaderProps = {
  // Legacy props from the previous design; no longer rendered.
  label?: string;
  description?: string;
  title: string;
};

export function SectionHeader({ title }: SectionHeaderProps) {
  return (
    <div className="mb-5 flex items-end justify-between gap-4 border-b border-line pb-1">
      <h2 className="font-display text-[34px] leading-none tracking-[-0.03em] text-foreground sm:text-[40px]">{title}</h2>
      <Link href="/#home" className="mb-1 shrink-0 text-[11px] text-muted hover:text-foreground">
        Back to top &#9650;
      </Link>
    </div>
  );
}
