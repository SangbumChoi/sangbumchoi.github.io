# Liquid 4 still calls Ruby's removed taint APIs when rendered on modern Ruby.
class Object
  def tainted?
    false
  end unless method_defined?(:tainted?)

  def untaint
    self
  end unless method_defined?(:untaint)
end
