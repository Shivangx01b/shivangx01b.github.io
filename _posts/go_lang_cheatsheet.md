---
date: 2020-07-25 22:55:45
layout: post
title: Go CheatSheet
description: Cheatsheet for golang 
image: https://i.ibb.co/zSg8TF1/Gocheat.png
optimized_image: https://i.ibb.co/zSg8TF1/Gocheat.png
category: Coding
tags:
  - coding
  - programming
  - go
author: shivangx01b
---

I decided to write something and not just waste my time in free time ¯\_(?)_/¯. So yeah ! I decided to write about Go as currently I'm learning it too :), So here we go !.

##BASICS

- Strings

```go
str := "Single Line"

str := `This is 
multiline string`
```

- Numbers

```go
num := 5          // int
num := 5.         // float64
num := 5 + 3i     // complex128
num := byte('s')  // byte (alias for uint8)

var u uint = 7        // uint (unsigned)
var p float32 = 22.7  // 32-bit float
```

- Arrays

```go
var numbers [5]int
numbers := [...]int{0, 0, 0, 0, 0}
```

-Slices

```go
slice := []int{2, 3, 4}

slice := []byte("Hello")
```

- Pointers

```go
func main () {
  b := *thisisapointer()
  fmt.Println("Value is", b)
}
 

func getPointer () (thisisapointer *int) {
  a := 234
  return &a
}
 

a := new(int)
*a = 234

```

- Type Conversions

```go
i := 2
f := float64(i)
u := uint(i)
```



