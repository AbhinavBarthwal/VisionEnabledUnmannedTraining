import React, { useEffect } from 'react'
import './LandingPage.css'

const LandingPage = ({ onContinue }) => {
  useEffect(() => {
    const timer = setTimeout(onContinue, 3000) // auto-transition in 3s
    return () => clearTimeout(timer)
  }, [])

  return (
    <div className="landing-container">
      <h1 className="veut-title">VEUT</h1>
      <p className="veut-subtitle">Vision-Enabled Unmanned Training</p>
    </div>
  )
}

export default LandingPage
