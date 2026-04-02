import { skillGroups } from "@/data/skills";
import { Container } from "@/components/ui/container";
import { Reveal } from "@/components/ui/reveal";
import { SectionHeader } from "@/components/section-header";

export function SkillsGrid() {
  return (
    <Reveal>
      <Container className="px-6 py-8 sm:px-8 sm:py-10 lg:px-10">
        <div className="space-y-8">
          <SectionHeader
            label="/ skills and tools"
            title="Skills and tools"
            description="The tools I use most often in research and technical work."
          />

          <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-5">
            {skillGroups.map((group) => (
              <article key={group.title} className="rounded-[26px] border border-line bg-white/[0.03] p-5">
                <p className="font-mono text-[11px] uppercase tracking-[0.24em] text-accent">{group.title}</p>
                <div className="mt-5 flex flex-wrap gap-2">
                  {group.items.map((item) => (
                    <span
                      key={item}
                      className="rounded-full border border-line bg-white/[0.03] px-3 py-1.5 text-sm text-slate-200"
                    >
                      {item}
                    </span>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </div>
      </Container>
    </Reveal>
  );
}
