/*
 * Copyright (c) 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstddef>

#define STDEXEC_AUTO_RETURN(...) \
  noexcept(noexcept(__VA_ARGS__))->decltype(__VA_ARGS__) { \
    return __VA_ARGS__; \
  }

/**/

namespace stdexec {
  template <class _Fn, class... _Args>
  using __tuple_applicable = __mbool<__callable<_Fn, _Args...>>;

  template <class _Fn, class... _Args>
  using __tuple_nothrow_apply_ = __mbool<__nothrow_callable<_Fn, _Args...>>;

  template <class... _Ts>
  struct __tuple {
    std::tuple<_Ts...> __vals;

    template <class _Fn, __decays_to<__tuple> _Self>
    static constexpr auto __apply(_Fn __fn, _Self&& __self) noexcept(
      __nothrow_callable<_Fn, __copy_cvref_t<_Self, _Ts>...>)
      -> __call_result_t<_Fn, __copy_cvref_t<_Self, _Ts>...> {
      return std::apply((_Fn&&) __fn, ((_Self&&) __self).__vals);
    }
  };

  template <class... _Ts>
  __tuple(_Ts...) -> __tuple<_Ts...>;

  template <>
  struct __tuple<> {
    template <class _Fn>
    static constexpr auto __apply(_Fn __fn, __tuple<>) STDEXEC_AUTO_RETURN(((_Fn&&) __fn)())
  };

  template <class _T0>
  struct __tuple<_T0> {
    _T0 __val0;

    template <class _Fn, __decays_to<__tuple> _Self>
    static constexpr auto __apply(_Fn __fn, _Self&& __self)
      STDEXEC_AUTO_RETURN(((_Fn&&) __fn)(((_Self&&) __self).__val0))
  };

  template <class _T0, class _T1>
  struct __tuple<_T0, _T1> {
    _T0 __val0;
    _T1 __val1;

    template <class _Fn, __decays_to<__tuple> _Self>
    static constexpr auto __apply(_Fn __fn, _Self&& __self)
      STDEXEC_AUTO_RETURN(((_Fn&&) __fn)(((_Self&&) __self).__val0, ((_Self&&) __self).__val1))
  };

  template <class _T0, class _T1, class _T2>
  struct __tuple<_T0, _T1, _T2> {
    _T0 __val0;
    _T1 __val1;
    _T2 __val2;

    template <class _Fn, __decays_to<__tuple> _Self>
    static constexpr auto __apply(_Fn __fn, _Self&& __self) STDEXEC_AUTO_RETURN(((
      _Fn&&) __fn)(((_Self&&) __self).__val0, ((_Self&&) __self).__val1, ((_Self&&) __self).__val2))
  };

  template <class _T0, class _T1, class _T2, class _T3>
  struct __tuple<_T0, _T1, _T2, _T3> {
    _T0 __val0;
    _T1 __val1;
    _T2 __val2;
    _T3 __val3;

    template <class _Fn, __decays_to<__tuple> _Self>
    static constexpr auto __apply(_Fn __fn, _Self&& __self) STDEXEC_AUTO_RETURN(((_Fn&&) __fn)(
      ((_Self&&) __self).__val0,
      ((_Self&&) __self).__val1,
      ((_Self&&) __self).__val2,
      ((_Self&&) __self).__val3))
  };

  template <class _Fn, class _Tuple>
  constexpr auto __tuple_apply(_Fn __fn, _Tuple&& __tup)
    STDEXEC_AUTO_RETURN(__tup.__apply((_Fn&&) __fn, (_Tuple&&) __tup));

  template <class _Tuple>
  extern const std::size_t __tuple_size;

  template <class... _Ts>
  inline constexpr std::size_t __tuple_size<__tuple<_Ts...>> = sizeof...(_Ts);

  template <std::size_t>
  struct __tuple_elem_fn_;

  template <>
  struct __tuple_elem_fn_<0u> {
    template <class _Tuple>
    using __f = decltype((__declval<_Tuple>().__val0));
  };

  template <>
  struct __tuple_elem_fn_<1u> {
    template <class _Tuple>
    using __f = decltype((__declval<_Tuple>().__val1));
  };

  template <>
  struct __tuple_elem_fn_<2u> {
    template <class _Tuple>
    using __f = decltype((__declval<_Tuple>().__val2));
  };

  template <>
  struct __tuple_elem_fn_<3u> {
    template <class _Tuple>
    using __f = decltype((__declval<_Tuple>().__val3));
  };

  template <std::size_t _Index, class _Tuple>
  using __tuple_elem = __minvoke<__tuple_elem_fn_<_Index>, _Tuple>;

  template <class _Tuple>
  using __tuple_elem0 = __tuple_elem<0u, _Tuple>;

  template <class _Tuple>
  using __tuple_elem1 = __tuple_elem<1u, _Tuple>;

  template <class _Tuple>
  using __tuple_elem2 = __tuple_elem<2u, _Tuple>;

  template <class _Tuple>
  using __tuple_elem3 = __tuple_elem<3u, _Tuple>;

} // namespace stdexec
