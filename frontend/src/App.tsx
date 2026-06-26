import { Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import Upload from './pages/Upload'
import History from './pages/History'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Upload />} />
        <Route path="/history" element={<History />} />
      </Route>
    </Routes>
  )
}
