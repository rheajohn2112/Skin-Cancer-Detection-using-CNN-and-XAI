import Navbar from "./components/Navbar"
import 'bootstrap/dist/css/bootstrap.min.css';
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import Upload from "./components/Upload";
import ErrorBoundary from "./components/Navbar";
function App() {

  const router = createBrowserRouter([
    {
path:"/upload",
element:<><Navbar /><Upload/></>,
errorElement: <ErrorBoundary /> }
  ])
  return (
    <>
      <RouterProvider router={router} />
    </>
  )
}

export default App
