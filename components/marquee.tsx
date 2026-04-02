const keywords = [
  "Computer Vision",
  "Robotic Perception",
  "Human-Centered AI",
  "Autonomous Driving Safety",
  "Multimodal Sensing",
  "Deep Learning",
  "PyTorch",
  "ROS2",
  "PhD Researcher",
];

const items = [...keywords, ...keywords];

export function Marquee() {
  return (
    <div className="overflow-hidden bg-[#3e2723] py-4">
      <div className="flex animate-marquee gap-0 whitespace-nowrap">
        {items.map((item, i) => (
          <span
            key={i}
            className="font-mono text-xs uppercase tracking-[0.22em] text-[#f4c9d6]"
          >
            {item}
            <span className="mx-8 opacity-40">✦</span>
          </span>
        ))}
      </div>
    </div>
  );
}
