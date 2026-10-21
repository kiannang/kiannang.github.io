import { skillGroups } from "@/data/skills";
import { Container } from "@/components/ui/container";
import { Panel } from "@/components/ui/panel";
import { SectionHeader } from "@/components/section-header";

export function SkillsGrid() {
  return (
    <Container id="skills">
      <SectionHeader title="Skills and tools." />

      <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-3">
        {skillGroups.map((group) => (
          <Panel key={group.title} title={group.title}>
            <ul className="space-y-1 text-[13px] leading-5">
              {group.items.map((item) => (
                <li key={item} className="flex gap-2">
                  <span className="text-accent">&rsaquo;</span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </Panel>
        ))}
      </div>
    </Container>
  );
}
