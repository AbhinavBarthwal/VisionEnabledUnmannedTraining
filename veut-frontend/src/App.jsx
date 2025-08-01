import React, { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import LandingPage from '../components/LandingPage'
import AnalysisPage from '../components/AnalysisPage'

function App() {
  const [showLanding, setShowLanding] = useState(true)

  return (
    <AnimatePresence>
      {showLanding ? (
        <motion.div
          key="landing"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <LandingPage onContinue={() => setShowLanding(false)} />
        </motion.div>
      ) : (
        <motion.div
          key="analysis"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <AnalysisPage />
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export default App
