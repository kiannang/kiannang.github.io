import Link from "next/link";
import { publications } from "@/data/publications";
import { Container } from "@/components/ui/container";
import { Panel } from "@/components/ui/panel";
import { SectionHeader } from "@/components/section-header";

export function PublicationCard() {
  const publication = publications[0];

  return (
    <Container id="publications">
      <SectionHeader title="Publications." />

      <Panel title="Featured Publication" aside={`${publication.venue} · ${publication.year}`}>
        <div className="grid gap-5 md:grid-cols-[1.4fr_1fr]">
          <div className="space-y-3">
            <h3 className="font-display text-[24px] leading-tight tracking-[-0.02em] text-foreground sm:text-[26px]">
              {publication.title}
            </h3>
            <p className="text-[12px] italic text-muted">{publication.authors}</p>
            <p className="text-[13px] leading-6">{publication.summary}</p>
          </div>

          <div className="space-y-3 self-start rounded-[8px] border border-line bg-panelStrong p-4 text-[12px]">
            <div>
              <p className="font-bold text-muted">Venue</p>
              <p className="mt-1 leading-5">{publication.fullVenue}</p>
            </div>
            <div>
              <p className="font-bold text-muted">Citation note</p>
              <p className="mt-1 leading-5">Available via DOI and institutional repository.</p>
            </div>
            <div className="flex flex-wrap gap-2 pt-1">
              <Link href={publication.links.paper} target="_blank" rel="noreferrer" className="aqua-btn-primary">
                View Paper
              </Link>
              <Link href={publication.links.doi} target="_blank" rel="noreferrer" className="aqua-btn">
                DOI
              </Link>
            </div>
          </div>
        </div>
      </Panel>
    </Container>
  );
}
