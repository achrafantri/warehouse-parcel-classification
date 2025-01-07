package com.example.achats.repository;

import com.example.achats.model.Categorie;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
/**
 * Repository bech nod5lou lel données categorie
 * <p>
 * kif kif sta3malna feha jpa bech CRUD (create read update delete) yet3mal automatique

 * </p>
 */
@Repository
public interface CategorieRepository extends JpaRepository<Categorie, Long> {
}
