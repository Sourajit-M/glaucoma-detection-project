import { Link, useLocation } from 'react-router-dom'
import { SiGithub } from 'react-icons/si'

const links = [
  { to: '/',          label: 'PREDICT'   },
  { to: '/dashboard', label: 'DASHBOARD' },
  { to: '/about',     label: 'ABOUT'     },
]

export default function Navbar() {
  const { pathname } = useLocation()
  return (
    <nav className="sticky top-0 z-50 glass-panel !rounded-none !border-t-0 !border-x-0 !border-b border-white/50 px-6 py-4 flex items-center gap-8 shadow-sm">
      <span style={{ fontFamily: 'var(--font-mono)' }}
        className="text-lg font-bold tracking-widest text-gradient mr-4 drop-shadow-sm">
        GLAUCOMADETECT
      </span>
      <div className="flex items-center gap-2 flex-1 justify-end mr-6">
        {links.map(l => (
          <Link key={l.to} to={l.to}
            style={{ fontFamily: 'var(--font-mono)' }}
            className={`text-xs font-semibold tracking-widest transition-all duration-300 px-4 py-2 rounded-lg
              ${pathname === l.to
                ? 'bg-blue-50 text-blue-600 shadow-sm'
                : 'text-gray-500 hover:text-blue-500 hover:bg-gray-50/50'}`}>
            {l.label}
          </Link>
        ))}
      </div>
      <a href="https://github.com/Sourajit-M/glaucoma-detection-project" target="_blank" rel="noreferrer"
        className="flex items-center gap-2 text-xs font-bold tracking-widest
          bg-gradient-to-r from-gray-800 to-black text-white px-5 py-2.5 rounded-xl hover:shadow-lg hover:-translate-y-0.5 transition-all duration-300"
        style={{ fontFamily: 'var(--font-mono)' }}>
        <SiGithub size={14} />
        GITHUB
      </a>
    </nav>
  )
}