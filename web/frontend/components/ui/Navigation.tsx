"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { href: "/", label: "Home" },
  { href: "/live", label: "Live", badge: true },
  { href: "/matches", label: "Matches" },
  { href: "/leaderboard", label: "Rankings" },
  { href: "/new-match", label: "New Match", highlight: true },
];

export function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="border-b border-arena-border bg-arena-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2">
            <span className="text-2xl">♟️</span>
            <span className="font-bold text-lg">Game Arena</span>
          </Link>

          {/* Nav Links */}
          <div className="flex items-center gap-1">
            {navItems.map((item) => {
              const isActive = pathname === item.href;
              const isHighlight = 'highlight' in item && item.highlight;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`
                    px-4 py-2 rounded-lg text-sm font-medium transition-colors
                    ${
                      isActive
                        ? "bg-arena-accent text-white"
                        : isHighlight
                        ? "bg-gradient-to-r from-arena-accent/20 to-purple-500/20 border border-arena-accent/50 text-arena-accent hover:bg-arena-accent hover:text-white"
                        : "text-gray-400 hover:text-white hover:bg-arena-border"
                    }
                  `}
                >
                  <span className="flex items-center gap-2">
                    {isHighlight && <span>+</span>}
                    {item.label}
                    {item.badge && (
                      <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                    )}
                  </span>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}

