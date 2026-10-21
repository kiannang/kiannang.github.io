import Link from "next/link";
import { Mail } from "lucide-react";
import { site } from "@/data/site";
import { Container } from "@/components/ui/container";

export function ContactSection() {
  return (
    <Container id="contact" className="pb-4 pt-10 text-center">
      <hr className="mb-8 border-line" />

      <h2 className="font-display text-[44px] leading-none tracking-[-0.035em] text-foreground sm:text-[54px]">Get in Touch.</h2>
      <p className="mx-auto mt-3 max-w-[460px] text-[13px] leading-6 text-muted">
        If you&apos;d like to talk about research, collaboration, or anything on the site, feel free to reach out.
      </p>

      <div className="mx-auto mt-5 flex max-w-[420px] items-center justify-center gap-2">
        <label htmlFor="contact-email" className="sr-only">
          Email address
        </label>
        <input
          id="contact-email"
          type="text"
          readOnly
          value={site.email}
          className="min-w-0 flex-1 border border-[#7a7a7a] bg-white px-2 py-[3px] text-[13px] shadow-[inset_0_1px_2px_rgba(0,0,0,0.25)]"
        />
        <Link href={`mailto:${site.email}`} className="aqua-btn-primary shrink-0">
          <Mail className="h-3.5 w-3.5" />
          Email Me
        </Link>
      </div>
    </Container>
  );
}
