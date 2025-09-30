'use client'

interface Props {
  warnings: string[]
}

export default function BiasWarnings({ warnings }: Props) {
  if (warnings.length === 0) return null

  const getWarningLevel = (warning: string): 'critical' | 'warning' | 'info' => {
    if (warning.includes('critically') || warning.includes('Critical')) return 'critical'
    if (warning.includes('below') || warning.includes('underrepresented')) return 'warning'
    return 'info'
  }

  const getWarningIcon = (level: string) => {
    switch (level) {
      case 'critical':
        return '🚨'
      case 'warning':
        return '⚠️'
      default:
        return 'ℹ️'
    }
  }

  const getWarningColor = (level: string) => {
    switch (level) {
      case 'critical':
        return 'bg-red-900 border-red-600 text-red-100'
      case 'warning':
        return 'bg-yellow-900 border-yellow-600 text-yellow-100'
      default:
        return 'bg-blue-900 border-blue-600 text-blue-100'
    }
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-2xl font-semibold text-white mb-4 flex items-center">
        Bias Warning System (T087c)
        <span className="ml-2 text-warning-red animate-pulse">●</span>
      </h2>
      <div className="space-y-3">
        {warnings.map((warning, index) => {
          const level = getWarningLevel(warning)
          return (
            <div
              key={index}
              className={`p-4 rounded-lg border ${getWarningColor(level)} transition-all hover:scale-[1.02]`}
            >
              <div className="flex items-start">
                <span className="text-2xl mr-3">{getWarningIcon(level)}</span>
                <div className="flex-1">
                  <p className="font-medium">{warning}</p>
                  {level === 'critical' && (
                    <p className="text-xs mt-2 opacity-75">
                      Immediate action required to maintain data quality
                    </p>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>
      <div className="mt-4 text-sm text-gray-400">
        <p>Warnings trigger when regions fall below 50% of target quotas</p>
      </div>
    </div>
  )
}