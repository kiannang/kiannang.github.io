"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { site } from "@/data/site";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/#education", label: "Education", id: "education" },
  { href: "/#publications", label: "Publications", id: "publications" },
  { href: "/#experience", label: "Experience", id: "experience" },
  { href: "/#teaching", label: "Teaching", id: "teaching" },
  { href: "/#contact", label: "Contact", id: "contact" },
];

const subNavItems = [
  { href: "/#awards", label: "Awards" },
  { href: "/#skills", label: "Skills" },
  { href: "/#activities", label: "Activities" },
  { href: "/resume/CV.pdf", label: "CV" },
  { href: site.links.github, label: "GitHub", external: true },
  { href: site.links.linkedin, label: "LinkedIn", external: true },
  { href: `mailto:${site.email}`, label: "Email" },
];

const tabBase =
  "relative -mb-px flex h-[30px] shrink-0 items-center justify-center rounded-t-[7px] border border-b-0 px-4 text-[14px]";
const tabIdle =
  "border-[#a98f96] bg-[linear-gradient(to_bottom,#ffffff_0%,#f6eef0_50%,#e9dde1_51%,#f3eaed_100%)] text-foreground hover:bg-[linear-gradient(to_bottom,#ffffff_0%,#fbe6ec_50%,#f3cfda_51%,#f8e3e9_100%)]";
const tabActive =
  "border-[#2a1a17] bg-[linear-gradient(to_bottom,#9c7a6f_0%,#6d4c41_48%,#3e2723_52%,#5d4037_100%)] font-bold text-[#fde7ee]";

export function Navbar() {
  const [activeSection, setActiveSection] = useState("home");

  useEffect(() => {
    const sections = ["home", ...navItems.map((item) => item.id)]
      .map((id) => document.getElementById(id))
      .filter((section): section is HTMLElement => Boolean(section));

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

        if (visible?.target.id) {
          setActiveSection(visible.target.id);
        }
      },
      {
        rootMargin: "-28% 0px -50% 0px",
        threshold: [0, 0.2, 0.45, 0.7],
      },
    );

    sections.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, []);

  return (
    <header className="pinstripe sticky top-0 z-50 pt-3">
      <nav aria-label="Sections" className="no-scrollbar mx-auto flex max-w-[800px] items-end gap-[3px] overflow-x-auto px-4">
        <Link
          href="/#home"
          aria-label="Kianna Ng, home"
          className={cn(tabBase, "w-[58px] px-0 font-display text-[19px] italic", activeSection === "home" ? tabActive : tabIdle)}
        >
          KN
        </Link>
        {navItems.map((item) => (
          <Link
            key={item.id}
            href={item.href}
            aria-current={activeSection === item.id ? "true" : undefined}
            className={cn(tabBase, "sm:min-w-[104px]", activeSection === item.id ? tabActive : tabIdle)}
          >
            {item.label}
          </Link>
        ))}
      </nav>

      <div className="aqua-bar border-x-0 shadow-[0_2px_4px_rgba(62,39,35,0.12)]">
        <div className="no-scrollbar mx-auto flex max-w-[800px] items-center justify-between gap-5 overflow-x-auto whitespace-nowrap px-4 py-[3px] text-[12px] sm:px-8">
          {subNavItems.map((item) => (
            <Link
              key={item.label}
              href={item.href}
              target={item.external ? "_blank" : undefined}
              rel={item.external ? "noreferrer" : undefined}
              className="text-foreground hover:underline"
            >
              {item.label}
            </Link>
          ))}
        </div>
      </div>
    </header>
  );
}
