type SectionHeaderProps = {
  label: string;
  title: string;
  description: string;
};

export function SectionHeader({ label, title, description }: SectionHeaderProps) {
  return (
    <div className="max-w-3xl space-y-4">
      <p className="font-mono text-xs uppercase tracking-[0.28em] text-accent">{label}</p>
      <div className="space-y-3">
        <h2 className="text-3xl font-semibold tracking-tight text-foreground sm:text-4xl">{title}</h2>
        <p className="max-w-2xl text-base leading-7 text-muted sm:text-lg">{description}</p>
      </div>
    </div>
  );
}
