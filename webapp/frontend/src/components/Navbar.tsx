import { Link, useLocation } from 'react-router-dom'

const links = [
  { to: '/',          label: 'Predict'   },
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/about',     label: 'About'     },
]

export default function Navbar() {
  const { pathname } = useLocation()
  return (
    <nav className="border-b border-gray-200 px-6 py-3 flex items-center gap-6">
      <span className="font-semibold text-gray-900 mr-4">GlaucomaDetect</span>
      {links.map(l => (
        <Link
          key={l.to}
          to={l.to}
          className={`text-sm ${pathname === l.to
            ? 'text-blue-600 font-medium'
            : 'text-gray-500 hover:text-gray-900'}`}
        >
          {l.label}
        </Link>
      ))}
    </nav>
  )
}