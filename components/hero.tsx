import Image from "next/image";
import Link from "next/link";
import {
  Bot,
  Camera,
  Car,
  Download,
  HeartHandshake,
  Layers,
  Radar,
  type LucideIcon,
} from "lucide-react";
import { awards } from "@/data/educationAwards";
import { site } from "@/data/site";
import { Container } from "@/components/ui/container";
import { cn } from "@/lib/utils";

const satelliteIcons: LucideIcon[] = [Camera, Bot, HeartHandshake, Layers, Car, Radar];

const satellites = site.heroTags.map((label, i) => ({ label, icon: satelliteIcons[i % satelliteIcons.length] }));

// Dotted connector lines on the 750x370 desktop stage, from each satellite icon to the window.
const connectors = [
  [182, 50, 250, 112],
  [182, 175, 250, 175],
  [182, 300, 250, 240],
  [568, 50, 500, 112],
  [568, 175, 500, 175],
  [568, 300, 500, 240],
];

const updated = new Date().toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });

function TrafficLight({ from, via, to }: { from: string; via: string; to: string }) {
  return (
    <span
      className="h-[11px] w-[11px] rounded-full border border-black/25"
      style={{ background: `radial-gradient(circle at 50% 30%, ${from} 0%, ${via} 55%, ${to} 100%)` }}
    />
  );
}

function HeadshotWindow({ className }: { className?: string }) {
  return (
    <div className={cn("aqua-box overflow-hidden rounded-[8px] shadow-[0_12px_32px_rgba(62,39,35,0.35)]", className)}>
      <div className="relative flex h-[22px] items-center gap-[6px] border-b border-[#a9a9a9] bg-[repeating-linear-gradient(to_bottom,#f2f2f2_0_1px,#e3e3e3_1px_2px)] px-2">
        <TrafficLight from="#ffc2bd" via="#e0443e" to="#a8231d" />
        <TrafficLight from="#ffe7a8" via="#e6a92b" to="#a8730d" />
        <TrafficLight from="#c9f5b5" via="#59b33c" to="#2c7a17" />
        <span className="absolute inset-x-0 text-center text-[11px] text-[#333]">kianna.jpg</span>
      </div>
      <div className="relative aspect-[3/4]">
        <Image
          src="/images/kianna_headshot.png"
          alt="Kianna Ng"
          fill
          className="object-cover object-top"
          sizes="240px"
          priority
        />
      </div>
    </div>
  );
}

function Satellite({ label, icon: Icon, reverse }: { label: string; icon: LucideIcon; reverse?: boolean }) {
  return (
    <div className={cn("flex items-center gap-3", reverse && "flex-row-reverse text-right")}>
      <span className="gel-tile h-[52px] w-[52px]">
        <Icon className="h-6 w-6" strokeWidth={1.75} />
      </span>
      <span className="text-[12px] font-bold leading-snug text-foreground">{label}</span>
    </div>
  );
}

export function Hero() {
  const award = awards[0];

  return (
    <Container id="home" className="pb-4 pt-8">
      {/* Desktop: the iBook-and-devices composition */}
      <div className="relative mx-auto hidden h-[370px] w-[750px] min-[800px]:block">
        <svg className="absolute inset-0" width="750" height="370" aria-hidden="true">
          {connectors.map(([x1, y1, x2, y2]) => (
            <line
              key={`${x1}-${y1}`}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="#8d6e63"
              strokeWidth="2"
              strokeDasharray="0.5 5"
              strokeLinecap="round"
            />
          ))}
        </svg>

        {satellites.slice(0, 3).map((item, i) => (
          <div key={item.label} className="absolute right-[570px] w-[200px]" style={{ top: 24 + i * 125 }}>
            <Satellite {...item} reverse />
          </div>
        ))}

        <HeadshotWindow className="absolute left-[255px] top-0 w-[240px]" />

        {satellites.slice(3).map((item, i) => (
          <div key={item.label} className="absolute left-[570px] w-[180px]" style={{ top: 24 + i * 125 }}>
            <Satellite {...item} />
          </div>
        ))}
      </div>

      {/* Smaller screens: window on top, research areas below */}
      <div className="min-[800px]:hidden">
        <HeadshotWindow className="mx-auto w-[220px]" />
        <div className="mx-auto mt-6 grid max-w-[560px] grid-cols-1 gap-3 min-[420px]:grid-cols-2 sm:grid-cols-3">
          {satellites.map((item) => (
            <Satellite key={item.label} {...item} />
          ))}
        </div>
      </div>

      <h1 className="mt-8 text-center font-display text-[46px] leading-none tracking-[-0.035em] text-foreground sm:text-[62px]">
        Hi, I&apos;m Kianna Ng.
      </h1>

      <p className="mx-auto mt-4 max-w-[580px] text-balance text-center text-[15px] leading-6 text-foreground">
        I&apos;m a PhD student at UC Merced, working on getting my degree and going day by day.
      </p>

      <div className="mt-5 flex flex-wrap justify-center gap-3">
        <Link href="/#publications" className="aqua-btn-primary">
          View Work
        </Link>
        <Link href="/resume/CV.pdf" className="aqua-btn">
          <Download className="h-3.5 w-3.5" />
          Download CV
        </Link>
        <Link href="/#contact" className="aqua-btn">
          Contact Me
        </Link>
      </div>

      <div className="mt-8 flex flex-wrap items-center justify-between gap-2 px-1 text-[11px] text-muted">
        <span className="inline-flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-[radial-gradient(circle_at_50%_30%,#c9f5b5,#59b33c_55%,#2c7a17)]" />
          Open to collaboration
        </span>
        <span>Last updated {updated}</span>
      </div>

      <Link
        href="/#awards"
        className="aqua-bar mt-1 flex items-stretch overflow-hidden rounded-[6px] text-[12px] hover:border-accent/60"
      >
        <span className="flex shrink-0 items-center border-r border-line bg-[linear-gradient(to_bottom,#fbe0e8_0%,#f2b5c7_50%,#eda3b9_51%,#f6cad7_100%)] px-4 py-1 font-bold">
          Latest News
        </span>
        <span className="flex items-center truncate px-3 py-1 font-bold">
          {award.title} ({award.year})
        </span>
      </Link>

    </Container>
  );
}
