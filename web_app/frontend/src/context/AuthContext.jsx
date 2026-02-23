import React, { createContext, useState, useContext, useEffect } from 'react'
import { supabase, setRememberMe } from '../supabaseClient'

const AuthContext = createContext()

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export const AuthProvider = ({ children }) => {
  const [session, setSession] = useState(null)
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Get initial session
    supabase.auth.getSession().then(({ data: { session: currentSession } }) => {
      setSession(currentSession)
      if (currentSession?.user) {
        setUser({
          email: currentSession.user.email,
          name: currentSession.user.user_metadata?.full_name || 'User',
        })
      }
      setLoading(false)
    })

    // Listen for auth state changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, currentSession) => {
        setSession(currentSession)
        if (currentSession?.user) {
          setUser({
            email: currentSession.user.email,
            name: currentSession.user.user_metadata?.full_name || 'User',
          })
        } else {
          setUser(null)
        }
      }
    )

    return () => subscription.unsubscribe()
  }, [])

  const login = async (email, password, rememberMe = true) => {
    try {
      // Set the persistence preference before signing in
      setRememberMe(rememberMe)

      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      })

      if (error) {
        return { success: false, error: error.message }
      }

      return { success: true }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  const register = async (name, email, password, rememberMe = true) => {
    try {
      // Set the persistence preference before signing up
      setRememberMe(rememberMe)

      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            full_name: name,
          },
        },
      })

      if (error) {
        return { success: false, error: error.message }
      }

      return { success: true }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  const changePassword = async (_oldPassword, newPassword) => {
    try {
      const { error } = await supabase.auth.updateUser({
        password: newPassword,
      })

      if (error) {
        return { success: false, error: error.message }
      }

      return { success: true, message: 'Password changed successfully' }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  const logout = async () => {
    await supabase.auth.signOut()
    setSession(null)
    setUser(null)
  }

  const value = {
    token: session?.access_token || null,
    user,
    login,
    register,
    logout,
    changePassword,
    loading,
    isAuthenticated: !!session,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
